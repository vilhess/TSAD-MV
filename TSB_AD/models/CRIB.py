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


class DataEmbedding_inverted(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.1):
        super().__init__()
        self.value_embedding = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):
        '''
        x: [batch_size, patch_num, var_num, patch_len]
        x_mark: None
        '''
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            # the potential to take covariates (e.g. timestamps) as tokens
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1)) # x_mark is time stamp

        # x: [batch_size, patch_num, var_num, model_dim]
        return self.dropout(x)

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

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

    def forward(self, x, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError("Only modes norm and denorm are supported.")
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

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
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(
                zip(self.attn_layers, self.conv_layers)
            ):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm_layer is not None:
            x = self.norm_layer(x)

        return x, attns


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
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)

        x = x + self.dropout(new_x)  # residual

        y = self.norm1(x)  # norm

        # MLP
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn  # residual


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

        out, attn = self.inner_attention(
            queries, keys, values, attn_mask, tau=tau, delta=delta
        )

        out = out.permute(0, 2, 1, 3).reshape(B, L, -1)
        out = self.out_projection(out)

        return out, attn

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

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

    
class CRIB_Encoder(nn.Module):
    def __init__(self, args, patch_num):
        super().__init__()
        self.args = args
        self.patch_num = patch_num

        self.softplus = nn.Softplus()

        self.enc_embedding_1 = DataEmbedding_inverted(
            input_size=args.d_model, hidden_size=args.d_model, dropout=args.dropout
        )  # out: [batch_size, patch_num, var_num, model_dim] # Linear-embedding

        self.enc_embedding_2 = TCNBlock(
            in_channel=args.patch_len,
            out_channel_list=[64, args.d_model],
            kernel_size=3,
            dropout=args.dropout,
        )

        self.encoder = TransformerEncoder(
            attn_layers=[
                EncoderLayer(
                    attention=AttentionLayer(
                        attention=Attention(
                            mask_flag=False,
                            scale=None,
                            attention_dropout=args.dropout,
                            output_attention=args.output_attn,
                        ),
                        model_dim=args.d_model,
                        heads_num=args.n_heads,
                    ),
                    model_dim=args.d_model,
                    dropout=args.dropout,
                    activation=args.activation,
                )
                for _ in range(args.n_layers_encoder)
            ],
            norm_layer=nn.LayerNorm(args.d_model),
        )

        self.projector = nn.Sequential(
            nn.Linear(
                in_features=self.patch_num * self.args.d_model,
                out_features=self.args.d_model,
            ),
            nn.ReLU(),  # 激活函数: ReLU
            nn.Linear(
                self.args.d_model, self.args.d_model * 2
            ), 
        )

    def forward(self, x_enc, x_mark=None):
        B, P, N, L = x_enc.shape  # [batch_size, patch_num, var_num, patch_len]

        x_enc = x_enc.permute(0, 3, 2, 1)
        enc_out = self.enc_embedding_2(
            x=x_enc
        )  # [batch_size, model_dim, var_num, patch_num ]
        enc_out = enc_out.permute(
            0, 3, 2, 1
        )  # [batch_size, patch_num, var_num, model_dim]

        enc_out = enc_out.reshape(
            B, -1, self.args.d_model
        )  # [batch_size, patch_num * var_num, model_dim]

        enc_out, attns = self.encoder(
            x=enc_out
        )  # [batch_size, patch_num * var_num, model_dim]

        enc_out_tmp = enc_out.reshape(B, P, N, -1).permute(0, 2, 1, 3).reshape(B, N, -1)
        mapped = self.projector(enc_out_tmp)

        # convert to distribution
        eps = (
            torch.ones_like(self.softplus(mapped[:, :, self.args.d_model :])) * 1e-9
        )  # For the numerical stability.
        loc = mapped[:, :, : self.args.d_model]
        scale = self.softplus(mapped[:, :, self.args.d_model :]) + eps

        distribution = MultivariateNormal(
            loc=loc, covariance_matrix=torch.diag_embed(scale)
        )

        return enc_out, attns, distribution

    
class CRIB_PredHead(nn.Module):
    def __init__(self, args, patch_num):
        super().__init__()
        self.args = args
        self.patch_num = patch_num

        self.prediction_1 = nn.Linear(
            in_features=self.patch_num * self.args.d_model, out_features=self.args.d_model
        )
        self.act_1 = nn.ReLU()
        self.prediction_2 = nn.Linear(
            in_features=self.args.d_model, out_features=1
        )

        self.prediction_3 = TCNBlock(
            in_channel=self.args.d_model,
            out_channel_list=[int(self.args.d_model / 2), int(self.args.d_model / 2)],
            kernel_size=3,
            dropout=self.args.dropout,
        )
        self.prediction_4 = nn.Linear(
            in_features=self.patch_num * int(self.args.d_model / 2),
            out_features=self.args.d_model,
        )
        self.act_2 = nn.ReLU()
        self.act_3 = nn.ReLU()
        self.prediction_5 = nn.Linear(
            in_features=self.args.d_model, out_features=1
        )

    def forward(self, x_pred, x_mark=None):
        B, _, _ = x_pred.shape  # [batch_size, patch_num*var_num, model_dim]

        x_pred = x_pred.reshape(
            B, -1, self.args.in_dim, self.args.d_model
        )  # [batch_size, patch_num, var_num, model_dim]

        x_pred = x_pred.permute(0, 2, 1, 3).reshape(
            B, self.args.in_dim, -1
        )  # [batch_size, var_num, patch_num*model_dim]
        x_pred = self.act_1(
            self.prediction_1(x_pred)
        )  # [batch_size, var_num, model_dim]
        x_pred = self.prediction_2(x_pred)  # [batch_size, var_num, pred_len]
        x_pred = x_pred.permute(0, 2, 1)  # [batch_size, pred_len, var_num]
        return x_pred

class CRIB(nn.Module):
    def __init__(self, seq_len, patch_len, d_model, n_heads, n_layers_encoder, dropout, activation, output_attn, feats=1):
        super().__init__()
        self.args = type("args", (), {})()
        self.args.ws = seq_len
        self.args.patch_len = patch_len
        self.args.d_model = d_model
        self.args.n_heads = n_heads
        self.args.n_layers_encoder = n_layers_encoder
        self.args.dropout = dropout
        self.args.activation = activation
        self.args.output_attn = output_attn 
        self.args.in_dim = feats

        self.seq_len = self.args.ws
        self.start_idx = 0
        if self.seq_len % self.args.patch_len != 0:
            print(f"Sequence length {self.seq_len} is not divisible by patch length {self.args.patch_len}. Padding will be applied.")
            self.start_idx = self.seq_len % self.args.patch_len
            self.seq_len -= self.start_idx

        self.patch_num = self.seq_len//self.args.patch_len

        self.enc_pos_emded = PositionalEmbedding(
            d_model=(self.args.patch_len + 1) // 2 * 2, max_len=5000
        )
        self.dec_pos_emded = PositionalEmbedding(d_model=self.args.d_model, max_len=5000)

        self.ini_embedding = nn.Linear(
            in_features=self.args.patch_len, out_features=self.args.d_model
        )

        self.encoder = CRIB_Encoder(args=self.args, patch_num=self.patch_num)

        self.predictor = CRIB_PredHead(args=self.args, patch_num=self.patch_num)

        self.revinlayer = RevIN(num_features=self.args.in_dim)

        self.prior=None

    def get_prior(self, prior_type="norm", device="cuda"):
        assert prior_type == "norm"
        if not self.prior:
            prior_loc = torch.zeros(self.args.d_model).to(device)
            prior_cov = torch.eye(self.args.d_model).to(device)
            self.prior = MultivariateNormal(loc=prior_loc, covariance_matrix=prior_cov)
        return self.prior

    def forward(self, x, test_flag=False):
        B, S, N = x.shape
        x = x[:, self.start_idx:, :]
        P, L = self.patch_num, self.args.patch_len
        x = rearrange(x, "b (pn pl) d -> b pn pl d", pn=self.patch_num, pl=self.args.patch_len).permute(0,1,3,2)
        x_1 = x

        # RevIN
        x_1 = x_1.permute(0, 1, 3, 2).reshape(B, P * L, N)
        x_1 = self.revinlayer(x_1, mode="norm")

        x_1 = x_1.reshape(B, P, L, N).permute(
            0, 1, 3, 2
        )  # [batch_size, patch_num, var_num, patch_len] --seq_len

        x_1 = x_1.reshape(B, P * N, L)

        x_1 = x_1 + self.enc_pos_emded(x_1)
        x_1 = x_1.reshape(B, P, N, L)

        noise = 0.01 * torch.normal(x_1.mean().item(), x_1.std().item(), x_1.shape).to(
            x.device
        )

        x_2 = x_1 + noise

        # prior distribution (Gaussian)
        pz = self.get_prior(prior_type="norm", device=x.device)

        # Encoder-Decoder
        enc_out_1, enc_attns_1, qz_x_1 = self.encoder(x_enc=x_1, x_mark=None)
        if test_flag:
            enc_out_2, enc_attns_2, qz_x_2 = None, None, None
        else:
            enc_out_2, enc_attns_2, qz_x_2 = self.encoder(x_enc=x_2, x_mark=None)
        if test_flag:
            z = qz_x_1.mean
        else:
            z = qz_x_1.rsample()

        ### ELBO with KL
        kl = torch.distributions.kl.kl_divergence(qz_x_1, pz)  # [M*K*BS, TL or d]
        kl = torch.where(torch.isfinite(kl), kl, torch.zeros_like(kl))
        kl = torch.sum(kl)
        preds = self.predictor(enc_out_1)
        # RevIN
        preds = self.revinlayer(preds, mode="denorm")
        return enc_out_1, enc_attns_1, enc_out_2, enc_attns_2, preds, kl

class CRIB_Detector(BaseDetector):
    def __init__(self,
                 seq_len = 25,
                 patch_len = 8, 
                 d_model = 32, 
                 n_heads = 4,
                 n_layers_encoder = 3,
                 dropout = 0.1,
                 activation = "relu",
                 output_attn = True,
                 lr = 1e-3,
                 IB_Weight = 1.0,
                 KL_Weight = 1e-6,
                 Consistency_Weight = 1,
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
        self.IB_Weight = IB_Weight
        self.KL_Weight = KL_Weight
        self.Consistency_Weight = Consistency_Weight
        self.max_train_mask_proba = max_train_mask_proba

        self.model = CRIB(
            seq_len-1, patch_len, d_model, n_heads, n_layers_encoder, dropout, activation, output_attn, feats=feats 
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
        mse = nn.MSELoss()
        
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

                enc_out_1, enc_attns_1, enc_out_2, enc_attns_2, preds, kl = self.model(context, test_flag=False)

                pred_loss = mae(preds, target)
                assert not torch.isnan(pred_loss), "Pred Loss is nan"

                behaviour_consistency_loss = mse(enc_out_1, enc_out_2)

                loss = self.IB_Weight * pred_loss + self.Consistency_Weight * behaviour_consistency_loss + self.KL_Weight * kl

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
                        enc_out_1, enc_attns_1, enc_out_2, enc_attns_2, preds, kl = self.model(context, test_flag=True)

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
        data[mask==1] = 0
    
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
                enc_out_1, enc_attns_1, enc_out_2, enc_attns_2, preds, kl = self.model(context, test_flag=True)
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