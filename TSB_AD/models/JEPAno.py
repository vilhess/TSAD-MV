from __future__ import division
from __future__ import print_function

import numpy as np
import math
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
import tqdm
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

from .base import BaseDetector
from ..utils.dataset import ReconstructDataset, ReconstructCombinedDataset
from ..utils.torch_utility import EarlyStoppingTorch, get_gpu

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.hidden_layer = nn.Linear(in_dim, hid_dim)
        self.output_layer = nn.Linear(hid_dim, out_dim)
        self.residual_layer = nn.Linear(in_dim, out_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        hid = self.act(self.hidden_layer(x))
        out = self.output_layer(hid)
        res = self.residual_layer(x)
        out = out + res
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert (
            d_model % n_heads == 0
        ), f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.WQ = nn.Linear(d_model, d_model)
        self.WK = nn.Linear(d_model, d_model)
        self.WV = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = dropout

        self.head_dim = d_model // n_heads
        self.n_heads = n_heads

        self.rope = RotaryEmbedding(dim=self.head_dim // 2)

    def forward(self, q):
        bs, context, dim = q.size()

        k = q
        v = q

        q = self.WQ(q).reshape(bs, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.WK(k).reshape(bs, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.WV(v).reshape(bs, -1, self.n_heads, self.head_dim).transpose(1, 2)

        q = self.rope.rotate_queries_or_keys(q)
        k = self.rope.rotate_queries_or_keys(k)

        values = nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )

        values = values.transpose(1, 2).reshape(bs, -1, dim)
        values = self.out_proj(values)
        return values


class FeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.1, multiple_of=256):
        super().__init__()

        hidden_dim = d_model * 4
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)

        self.act = nn.SiLU()
        self.dp = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w2(self.act(self.w1(x)) * self.w3(x))
        return self.dp(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(
            d_model=d_model, n_heads=n_heads, dropout=dropout
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model=d_model, dropout=dropout)

    def forward(self, x):
        out_attn = self.attn(self.ln1((x)))
        x = x + out_attn
        out = x + self.ff(self.ln2(x))
        return out


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=d_model, n_heads=n_heads, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class Encoder(nn.Module):
    def __init__(
        self,
        patch_len,
        d_model,
        n_heads,
        n_layers_encoder,
        dropout=0.1,
    ):
        super().__init__()

        self.patch_len = patch_len

        self.proj_embedding = ResidualBlock(
            in_dim=2*patch_len, hid_dim=2 * d_model, out_dim=d_model, dropout=dropout
        )
        self.dp = nn.Dropout(dropout)
        self.transformer_encoder = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers_encoder,
            dropout=dropout,
        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                m.bias.data.fill_(0.0)
                m.weight.data.fill_(1.0)

    def forward(self, x):
        assert x.dim() == 2, "Input tensor must be (batch, patch_num*patch_len)"
        bs, ws = x.size()

        x = rearrange(
            x, "b (pn pl) -> b pn pl", pl=self.patch_len
        )  # Reshape to (bs, patch_num, patch_len)
        mask_pos = torch.isnan(x)

        x = torch.cat([x, mask_pos.float()], dim=-1)

        x = self.proj_embedding(x)  # bs, pn, d_model
        x = self.dp(x)
        x = self.transformer_encoder(x)  # bs, pn, d_model

        return x

    @torch.inference_mode()
    def get_encoding(self, x):
        assert x.dim() == 3, "Input tensor must be (batch, seq_len, feats)"
        bs, seq_len, feats = x.size()

        x = x.transpose(1, 2)  # (batch, feats, seq_len)
        x = rearrange(x, "b f s -> (b f) s")
        encoded = self(x)
        encoded = encoded[:, -1, :]
        encoded = rearrange(encoded, "(b f) d -> b f d", f=feats)
        encoded = rearrange(encoded, "b f d -> b (f d)")
        return encoded

class Predictor(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        n_layers_predictor,
        dropout=0.1,
    ):
        super().__init__()

        self.transformer_encoder = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers_predictor,
            dropout=dropout,
        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                m.bias.data.fill_(0.0)
                m.weight.data.fill_(1.0)

    def forward(self, x):
        x = self.transformer_encoder(x)  # bs, pn, d_model
        return x

class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)"""

    def __init__(self, knots=17, num_proj=1024, device=None):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32, device=device)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32, device=device)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0).to(device)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean() # average over projections and time

def get_bank_embedding(encoder, dataloader, num_cores=128, dim_embedding=64, device=None, mask_ratios=None):
    if mask_ratios is None:
        mask_ratios = [0.0, 0.1, 0.2, 0.3, 0.4]

    all_embeddings = []
    with torch.inference_mode():
        for batch, _ in dataloader:
            if device is not None:
                batch = batch.to(device)
            
            # Generate embeddings for each masking ratio
            for ratio in mask_ratios:
                if ratio > 0:
                    # Create random mask
                    mask = torch.rand_like(batch) < ratio
                    masked_batch = batch.clone()
                    masked_batch[mask] = 0
                else:
                    masked_batch = batch
                
                embeddings = encoder.get_encoding(masked_batch)
                all_embeddings.append(embeddings.cpu())
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_embeddings = torch.nn.functional.normalize(all_embeddings, dim=-1, p=2)

    n_samples = all_embeddings.size(0)
    embedding_dim = all_embeddings.size(1)

    pca = None
    if dim_embedding < embedding_dim:
        print(f"Applying PCA to reduce dimensionality from {embedding_dim} to {dim_embedding}")
        pca = PCA(n_components=dim_embedding)
        all_embeddings = torch.tensor(pca.fit_transform(all_embeddings.numpy()), dtype=all_embeddings.dtype)

    if num_cores>n_samples:
        print(f"Number of cores ({num_cores}) is greater than number of samples ({n_samples}), returning all embeddings without clustering.")
        return all_embeddings, pca
    
    kmeans = MiniBatchKMeans(
        n_clusters=num_cores,
        init='k-means++',
        random_state=42,
        batch_size=max(8192, num_cores),   
        max_iter=50,                       
        n_init=1,                          
        reassignment_ratio=0.01
    )
    kmeans.fit(all_embeddings.numpy())
    centers = torch.tensor(kmeans.cluster_centers_, dtype=all_embeddings.dtype)  
    distances = torch.cdist(all_embeddings, centers, p=2)   
    core_indices = torch.argmin(distances, dim=0)   
    bank_embedding = all_embeddings[core_indices]
    return bank_embedding, pca

def compute_anomaly_score(encoder, test_loader, bank_embedding, pca, top_k=3, device=None):
    assert bank_embedding.size(0) >= top_k, f"top_k ({top_k}) must be less than or equal to the number of bank embeddings ({bank_embedding.size(0)})"
    all_scores = []
    with torch.inference_mode():
        for batch, _ in test_loader:
            if device is not None:
                batch = batch.to(device)
            embeddings = encoder.get_encoding(batch)
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1, p=2)
            if pca is not None:
                embeddings = torch.tensor(pca.transform(embeddings.cpu().numpy()), dtype=embeddings.dtype).to(embeddings.device)
            cos_similarity = torch.matmul(embeddings, bank_embedding.T)
            topk_values, _ = torch.topk(cos_similarity, k=top_k, dim=-1, largest=True)
            dist = 1 - topk_values
            scores = dist.mean(dim=-1)
            all_scores.append(scores.cpu())
    all_scores = torch.cat(all_scores, dim=0)
    return all_scores.numpy()

class JEPAno(BaseDetector):
    def __init__(self,
                 win_size = 64,
                 patch_len=8,
                 d_model=128,
                 n_heads=4,
                 n_layers_encoder=2,
                 n_layers_predictor=2,
                 dropout=0.05,
                 max_mask_proba=0.3,
                 epochs=100,

                 feats = 1,
                 batch_size = 128,
                 ):
        super().__init__()

        self.__anomaly_score = None

        self.cuda = True
        self.device = get_gpu(self.cuda)

        self.win_size = win_size
        self.d_model = d_model
        self.dim_embedding = d_model * feats
        self.batch_size = batch_size
        self.feats = feats
        self.max_mask_proba = max_mask_proba
        self.n_heads = n_heads
        self.n_layers_predictor = n_layers_predictor
        self.epochs = epochs
        self.dropout = dropout

        self.encoder = Encoder(
            patch_len=patch_len,
            d_model=d_model,
            n_heads=n_heads,
            n_layers_encoder=n_layers_encoder,
            dropout=dropout
        ).to(self.device)

    def fit(self, data):

        tsTrain = data

        train_loader = DataLoader(
            dataset=ReconstructDataset(tsTrain, window_size=self.win_size),
            batch_size=self.batch_size,
            shuffle=True
        )

        max_mask_proba = self.max_mask_proba

        sigreg_loss = SIGReg(device=self.device)
        mse_loss = nn.MSELoss()
        lamb_sigreg = 0.09
        predictor = Predictor(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers_predictor=self.n_layers_predictor,
            dropout=self.dropout
        ).to(self.device)
        optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(predictor.parameters()),
            lr=1e-3,
            weight_decay=1e-5,
        )
        self.optimizer = optimizer
        self.lambda_sigreg = lamb_sigreg
        self.lambda_pred = 10

        for epoch in range(1, self.epochs + 1):
            self.encoder.train(mode=True)
            predictor.train(mode=True)
            avg_loss = 0
            loop = tqdm.tqdm(
                enumerate(train_loader), total=len(train_loader), leave=True
            )
            for idx, (x, _) in loop:                
                if torch.isnan(x).any() or torch.isinf(x).any():
                    print("Input data contains nan or inf")
                    x = torch.nan_to_num(x)
                self.optimizer.zero_grad()

                x = x.to(self.device)
                x = x.transpose(1, 2)
                x = rearrange(x, "b f s -> (b f) s")
                bs = x.shape[0]

                mask_proba = torch.rand(x.shape, device=x.device) * self.max_mask_proba
                mask = torch.bernoulli(mask_proba).bool()
                x_masked = x.masked_fill(mask, 0)
                encoded = self.encoder(x_masked)
                pred = predictor(encoded)
                
                sigreg_loss_value = sigreg_loss(encoded.transpose(0, 1))
                pred_loss_value = mse_loss(pred[:, :-1], encoded[:, 1:].detach())

                loss = self.lambda_pred * pred_loss_value + self.lambda_sigreg * sigreg_loss_value

                loss.backward(retain_graph=True)

                self.optimizer.step()
                avg_loss += loss.cpu().item()
                loop.set_description(f"Training Epoch [{epoch}/{self.epochs}]")
                loop.set_postfix({"loss": avg_loss / (idx + 1), "pred_loss": pred_loss_value.cpu().item(), "sigreg_loss": sigreg_loss_value.cpu().item()})

        print("Extracting bank embeddings...")
        bank_embedding, pca = get_bank_embedding(self.encoder, train_loader, num_cores=500, dim_embedding=self.dim_embedding, device=self.device)
        print("Bank embeddings extracted.")
        self.bank_embedding = bank_embedding.to(self.device)
        self.pca = pca

    def decision_function(self, all_datas):
        data = all_datas["data_missing"]
        true_data = all_datas["data"] # last timestep per window never contains missing value 
    
        test_loader = DataLoader(
            dataset=ReconstructCombinedDataset(data, true_data, window_size=self.win_size, normalize=False),
            batch_size=self.batch_size,
            shuffle=False,
        )

        self.encoder.eval()
        scores = compute_anomaly_score(self.encoder, test_loader, self.bank_embedding, self.pca, top_k=3, device=self.device)

        self.__anomaly_score = scores

        if self.__anomaly_score.shape[0] < len(data):
            self.__anomaly_score = np.array([self.__anomaly_score[0]]*math.ceil((self.win_size-1)/2) + 
                        list(self.__anomaly_score) + [self.__anomaly_score[-1]]*((self.win_size-1)//2))
        
        return self.__anomaly_score

    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score

    def param_statistic(self, save_file):
        pass
