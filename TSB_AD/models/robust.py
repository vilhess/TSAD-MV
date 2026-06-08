from __future__ import division
from __future__ import print_function

import numpy as np
import math
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn.utils.parametrizations import weight_norm
from einops import rearrange
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
import tqdm

from .base import BaseDetector
from ..utils.dataset import ReconstructDataset, ReconstructCombinedDataset
from ..utils.torch_utility import EarlyStoppingTorch, get_gpu


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1), :x.size(2)]
    
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """Reversible Instance Normalization for Accurate Time-Series Forecasting
               against Distribution Shift, ICLR2021.

        Parameters
        ----------
        num_features: int, the number of features or channels.
        eps: float, a value added for numerical stability, default 1e-5.
        affine: bool, if True(default), RevIN has learnable affine parameters.
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str, mask=None):
        if mode == "norm":
            self._get_statistics(x, mask=mask) # was okay without considering mask
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError("Only modes norm and denorm are supported.")
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x, mask=None):
        dim2reduce = tuple(range(1, x.ndim - 1))
        x_tmp = x.clone()
        if mask is not None:
            x_tmp = x_tmp.masked_fill(mask.bool(), torch.nan)

        sum = torch.nansum(x_tmp, dim=dim2reduce, keepdim=True).detach()
        count = torch.sum(~torch.isnan(x_tmp), dim=dim2reduce, keepdim=True).detach()
        self.mean = (sum / count).detach()
        self.stdev = torch.sqrt((torch.nansum((x_tmp - self.mean) ** 2, dim=dim2reduce, keepdim=True) / count)+self.eps).detach()


    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[..., : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride, dilation, padding, dropout
    ):
        super().__init__()
        self.conv1 = weight_norm(
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=(1, kernel_size),
                stride=stride,
                padding=(0, padding),
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv2d(
                in_channels=out_channel,
                out_channels=out_channel,
                kernel_size=(1, kernel_size),
                stride=stride,
                padding=(0, padding),
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = (
            nn.Conv2d(in_channel, out_channel, 1) if in_channel != out_channel else None
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return out + res


class TCNBlock(nn.Module):
    def __init__(self, in_channel, out_channel_list, kernel_size, dropout):
        super().__init__()
        layers = []

        for i in range(len(out_channel_list)):
            dilation_size = 2**i
            in_channel = in_channel if i == 0 else out_channel_list[i - 1]
            out_channel = out_channel_list[i]
            layers += [
                TemporalBlock(
                    in_channel=in_channel,
                    out_channel=out_channel,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TransformerEncoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = (
            nn.ModuleList(conv_layers) if conv_layers is not None else None
        )
        self.norm_layer = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(
                zip(self.attn_layers, self.conv_layers)
            ):
                delta = delta if i == 0 else None
                x = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
            x = self.attn_layers[-1](x, attn_mask=attn_mask, tau=tau, delta=delta)
        else:
            for attn_layer in self.attn_layers:
                x = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)

        if self.norm_layer is not None:
            x = self.norm_layer(x)

        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, model_dim, dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = 4 * model_dim
        self.attention = attention
        self.conv1 = nn.Conv1d(
            in_channels=model_dim, out_channels=d_ff, kernel_size=1
        )  # equal to MLP
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=model_dim, kernel_size=1)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x = self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)

        x = x + self.dropout(new_x)  # residual

        y = self.norm1(x)  # norm

        # MLP
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y)


class AttentionLayer(nn.Module):
    def __init__(self, attention, model_dim, heads_num, keys_dim=None, values_dim=None):
        super().__init__()
        keys_dim = keys_dim or (model_dim // heads_num)
        values_dim = values_dim or (model_dim // heads_num)

        self.inner_attention = attention
        self.query_projection = nn.Linear(model_dim, keys_dim * heads_num)
        self.key_projection = nn.Linear(model_dim, keys_dim * heads_num)
        self.value_projection = nn.Linear(model_dim, values_dim * heads_num)
        self.out_projection = nn.Linear(values_dim * heads_num, model_dim)
        self.heads_num = heads_num

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.heads_num

        queries = self.query_projection(queries).view(B, L, H, -1)

        keys = self.query_projection(keys).view(B, S, H, -1)
        values = self.query_projection(values).view(B, S, H, -1)

        out = self.inner_attention(
            queries, keys, values, attn_mask, tau=tau, delta=delta
        )

        out = out.permute(0, 2, 1, 3).reshape(B, L, -1)
        out = self.out_projection(out)

        return out
    
class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

class Attention(nn.Module):
    def __init__(
        self, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False
    ):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape  # [batch_size, seq_len, hidden_size, embed_size]
        _, S, _, D = values.shape  # [batch_size, pred_len, hidden_size, embed_size]
        scale = self.scale or 1.0 / math.sqrt(E)

        scores = torch.einsum(
            "blhe,bshe->bhls", queries, keys
        )

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return V.contiguous()
class CRIB_Encoder(nn.Module):
    def __init__(self, patch_num, patch_len, d_model, n_heads, n_layers_encoder, dropout, activation):
        super().__init__()
        self.patch_num = patch_num
        self.patch_len = patch_len
        self.d_model = d_model

        self.softplus = nn.Softplus()
        
        self.enc_embedding_2 = TCNBlock(
            in_channel=patch_len,
            out_channel_list=[64, d_model],
            kernel_size=3,
            dropout=dropout,
        )

        self.encoder = TransformerEncoder(
            attn_layers=[
                EncoderLayer(
                    attention=AttentionLayer(
                        attention=Attention(
                            mask_flag=False,
                            scale=None,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        model_dim=d_model,
                        heads_num=n_heads,
                    ),
                    model_dim=d_model,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(n_layers_encoder)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )

        self.projector = nn.Sequential(
            nn.Linear(
                in_features=self.patch_num * self.d_model,
                out_features=self.d_model,
            ),
            nn.ReLU(),  # 激活函数: ReLU
            nn.Linear(
                self.d_model, self.d_model * 2
            ), 
        )

    def forward(self, x_enc):
        B, P, N, L = x_enc.shape  # [batch_size, patch_num, var_num, patch_len]

        x_enc = x_enc.permute(0, 3, 2, 1)
        enc_out = self.enc_embedding_2(
            x=x_enc
        )  # [batch_size, model_dim, var_num, patch_num ]
        enc_out = enc_out.permute(
            0, 3, 2, 1
        )  # [batch_size, patch_num, var_num, model_dim]

        enc_out = enc_out.reshape(
            B, -1, self.d_model
        )  # [batch_size, patch_num * var_num, model_dim]

        enc_out = self.encoder(
            x=enc_out
        )  # [batch_size, patch_num * var_num, model_dim]
        return enc_out
    
class CRIB_PredHead(nn.Module):
    def __init__(self, patch_num, d_model, feats):
        super().__init__()
        self.patch_num = patch_num
        self.d_model = d_model
        self.feats = feats

        self.prediction_1 = nn.Linear(
            in_features=self.patch_num * self.d_model, out_features=self.d_model
        )
        self.act_1 = nn.ReLU()
        self.prediction_2 = nn.Linear(
            in_features=self.d_model, out_features=1 # predict the next timestep value for each variable
        )

    def forward(self, x_pred):
        B, _, _ = x_pred.shape  # [batch_size, patch_num*var_num, model_dim]

        x_pred = x_pred.reshape(
            B, -1, self.feats, self.d_model
        )  # [batch_size, patch_num, var_num, model_dim]
        x_pred = x_pred.permute(0, 2, 1, 3).reshape(
            B, self.feats, -1
        )  # [batch_size, var_num, patch_num*model_dim]
        x_pred = self.act_1(
            self.prediction_1(x_pred)
        )  # [batch_size, var_num, model_dim]
        x_pred = self.prediction_2(x_pred)  # [batch_size, var_num, 1]
        x_pred = x_pred.permute(0, 2, 1)  # [batch_size, 1, var_num]
        return x_pred
    
class FusionMaskValue(nn.Module):
    def __init__(self, patch_len, dropout=0.1, multiple_of=32):
        super().__init__()

        hidden_dim = patch_len*4
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(2*patch_len, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, patch_len, bias=False)
        self.w3 = nn.Linear(2*patch_len, hidden_dim, bias=False)

        self.act = nn.SiLU()
        self.dp = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w2(self.act(self.w1(x)) * self.w3(x))
        return self.dp(x)

class CRIB(nn.Module):
    def __init__(self, seq_len, patch_len, d_model, n_heads, n_layers_encoder, dropout, activation, feats):
        super().__init__()

        self.seq_len = seq_len
        self.patch_len = patch_len
        self.d_model = d_model
        self.feats = feats
        self.start_idx = 0
        if self.seq_len % patch_len != 0:
            self.start_idx = self.seq_len % patch_len
            self.seq_len -= self.start_idx

        self.patch_num = self.seq_len//patch_len

        self.enc_pos_emded = PositionalEmbedding(
            d_model=(patch_len + 1) // 2 * 2, max_len=5000
        )

        self.encoder = CRIB_Encoder(patch_num=self.patch_num, patch_len=patch_len, d_model=d_model, n_heads=n_heads, n_layers_encoder=n_layers_encoder, dropout=dropout, activation=activation)

        self.predictor = CRIB_PredHead(patch_num=self.patch_num, d_model=d_model, feats=self.feats)

        self.revinlayer = RevIN(num_features=self.feats)
        self.fusion = FusionMaskValue(patch_len=patch_len)

    def forward(self, x, mask=None):
        assert mask.shape == x.shape, "mask shape should be the same as x shape"

        B, S, N = x.shape
        P, L = self.patch_num, self.patch_len
        x = x[:, self.start_idx:, :]
        mask = mask[:, self.start_idx:, :]

        # RevIN
        x_1 = self.revinlayer(x, mode="norm", mask=mask)

        x_1 = x_1.reshape(B, P, L, N).permute(
            0, 1, 3, 2
        )  # [batch_size, patch_num, var_num, patch_len] 
        x_1 = x_1.reshape(B, P * N, L)

        x_1 = x_1 + self.enc_pos_emded(x_1)
        x_1 = x_1.reshape(B, P, N, L)
        mask = mask.reshape(B, P, L, N).permute(0, 1, 3, 2)  # [batch_size, patch_num, var_num, patch_len]
        x_1 = torch.stack([x_1, mask], dim=-1).flatten(start_dim=-2)
        x_1 = self.fusion(x_1)

        # Encoder-Decoder
        enc_out_1 = self.encoder(x_enc=x_1)
        preds = self.predictor(enc_out_1)

        # RevIN
        preds = self.revinlayer(preds, mode="denorm")
        return preds, enc_out_1

class CRIB_Detector(BaseDetector):
    def __init__(self,
                 seq_len = 65,
                 patch_len = 8, 
                 d_model = 128, 
                 n_heads = 4,
                 n_layers_encoder = 4,
                 dropout = 0.1,
                 activation = "relu",
                 lr = 1e-3,
                 max_train_mask_proba = 0.5,

                 feats = 1,
                 batch_size = 256,
                 epochs = 25,
                 patience = 3,
                 validation_size=0.2
                 ):
        super().__init__()

        self.__anomaly_score = None

        self.cuda = True
        self.device = get_gpu(self.cuda)

        self.win_size = seq_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.feats = feats
        self.validation_size = validation_size
        self.lr = lr
        self.max_train_mask_proba = max_train_mask_proba

        self.model = CRIB(
            seq_len-1, patch_len, d_model, n_heads, n_layers_encoder, dropout, activation, feats=feats 
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0)

        self.early_stopping = EarlyStoppingTorch(None, patience=patience)

    def fit(self, data):

        tsTrain = data[:int((1-self.validation_size)*len(data))]
        tsValid = data[int((1-self.validation_size)*len(data)):]

        train_loader = DataLoader(
            dataset=ReconstructDataset(tsTrain, window_size=self.win_size, normalize=False),
            batch_size=self.batch_size,
            shuffle=True
        )
        
        valid_loader = DataLoader(
            dataset=ReconstructDataset(tsValid, window_size=self.win_size, normalize=False),
            batch_size=self.batch_size,
            shuffle=False
        )
        mae = nn.L1Loss()
        
        for epoch in range(1, self.epochs + 1):
            self.model.train(mode=True)
            avg_loss = 0
            loop = tqdm.tqdm(
                enumerate(train_loader), total=len(train_loader), leave=True
            )
            for idx, (x, _) in loop:      
                assert not torch.isnan(x).any() and not torch.isinf(x).any(), "Input data contains nan or inf"
                self.optimizer.zero_grad()
                x = x.to(self.device)
                bs = x.shape[0]

                mask_proba = self.max_train_mask_proba * np.random.rand()

                context = x[:, :-1, :]
                target = x[:, -1:, :]
                mask = torch.bernoulli(torch.ones_like(context)*mask_proba).to(context.device)
                context = context.masked_fill(mask.bool(), 0)

                preds, enc_out_1 = self.model(context, mask=mask)

                loss = mae(preds, target)
                assert not torch.isnan(loss), "Loss is nan"

                loss.backward(retain_graph=True)

                self.optimizer.step()
                avg_loss += loss.cpu().item()
                loop.set_description(f"Training Epoch [{epoch}/{self.epochs}]")
                loop.set_postfix(loss=loss.item(), avg_loss=avg_loss / (idx + 1))

            if torch.isnan(loss):
                print(f"Loss is nan at epoch {epoch}")

            if len(valid_loader) > 0:
                self.model.eval()
                avg_loss_val = 0
                loop = tqdm.tqdm(
                    enumerate(valid_loader), total=len(valid_loader), leave=True
                )
                with torch.no_grad():
                    for idx, (x, _) in loop:      

                        assert not torch.isnan(x).any() and not torch.isinf(x).any(), "Input data contains nan or inf"

                        x = x.to(self.device)
                        # x = x.unsqueeze(-1)
                        bs = x.shape[0]
                        
                        context = x[:, :-1, :]
                        target = x[:, -1:, :]

                        mask_proba = self.max_train_mask_proba * np.random.rand()
                        mask = torch.bernoulli(torch.ones_like(context)*mask_proba).to(context.device)
                        context = context.masked_fill(mask.bool(), 0)
                        preds, enc_out_1 = self.model(context, mask=mask)

                        loss = ((preds - target)**2).flatten(start_dim=1).mean(dim=1).mean()

                        avg_loss_val += loss.cpu().item()
                        loop.set_description(f"Validation Epoch [{epoch}/{self.epochs}]")
                        loop.set_postfix(loss=loss.item(), avg_loss_val=avg_loss_val / (idx + 1))

            if len(valid_loader) > 0:
                avg_loss = avg_loss_val / len(valid_loader)
            else:
                avg_loss = avg_loss / len(train_loader)
            self.early_stopping(avg_loss, self.model)
            if self.early_stopping.early_stop:
                print("   Early stopping<<<")
                break

    def decision_function(self, all_datas):

        data_for_test = all_datas["data_missing"].copy()
        data_for_test = np.nan_to_num(data_for_test, nan=0.0)  # safe fallback
        data = data_for_test
        true_data = all_datas["data"] # last timestep per window never contains missing value 
        mask = all_datas["mask"]
        data[mask==1] = torch.nan
    
        test_loader = DataLoader(
            dataset=ReconstructCombinedDataset(data, true_data, window_size=self.win_size, normalize=False),
            batch_size=self.batch_size,
            shuffle=False,
        )

        self.model.eval()
        scores = []
        loop = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), leave=True)
        with torch.no_grad():
            for idx, (x, _) in loop:
                x = x.to(self.device) # bs, win_size, feats
                bs = x.shape[0]

                is_padded=False
                if bs == 1:
                    x = x.repeat(2, 1, 1)
                    is_padded=True

                context = x[:, :-1, :]
                target = x[:, -1:, :]
                context_mask = torch.isnan(context)
                context = context.masked_fill(context_mask, 0)
                preds, enc_out_1 = self.model(context, mask=context_mask)
                loss = ((preds - target)**2).flatten(start_dim=1).mean(dim=1)   
                
                if is_padded:
                    loss = loss[:1]
                scores.append(loss.cpu())

        scores = torch.cat(scores, dim=0)
        scores = scores.numpy()

        self.__anomaly_score = scores

        if self.__anomaly_score.shape[0] < len(data):
            self.__anomaly_score = np.array([self.__anomaly_score[0]]*math.ceil((self.win_size-1)/2) + 
                        list(self.__anomaly_score) + [self.__anomaly_score[-1]]*((self.win_size-1)//2))
        
        return self.__anomaly_score

    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score

    def param_statistic(self, save_file):
        pass