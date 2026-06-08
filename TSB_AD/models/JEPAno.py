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


def fill_nan_with_last_observed(x):
    bs, pn, pl = x.size()
    x = rearrange(x, "b pn pl -> b (pn pl)")
    valid_mask = ~torch.isnan(x)
    x_temp = torch.where(valid_mask, x, torch.zeros_like(x))
    seq_indices = torch.arange(x.size(-1), device=x.device).unsqueeze(0)
    valid_indices = torch.where(
        valid_mask, seq_indices, torch.tensor(-1, device=x.device)
    )
    last_valid_idx = torch.cummax(valid_indices, dim=-1)[0]
    x = x_temp.gather(-1, torch.clamp(last_valid_idx, min=0))
    x = rearrange(x, "b (pn pl) -> b pn pl", pn=pn)
    return x

class CausalRevIN(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.cached_mean = None
        self.cached_std = None

    def forward(self, x, mode):
        assert x.dim() == 3, "Input tensor must be (batch, n_patches, patch_len)"

        x64 = x.double()

        if mode == "norm":
            mean, std = self._get_statistics(x64)
            self.cached_mean, self.cached_std = mean, std
            out = (x64 - mean) / std
            out = torch.asinh(out)

            nan_idx = out.isnan()
            if nan_idx.any():
                out = fill_nan_with_last_observed(out)

        elif mode == "denorm":
            assert (
                self.cached_mean is not None and self.cached_std is not None
            ), "Call forward(..., 'norm') before 'denorm'"
            out = torch.sinh(x64) * self.cached_std + self.cached_mean

        else:
            raise NotImplementedError(f"Mode '{mode}' not implemented.")

        return out.float()

    def _get_statistics(self, x):
        """
        Numerically stable mean and variance computation using
        incremental mean and variance along the patch dimension.
        x: (B, P, L) float64
        Returns: mean, std (both (B, P, 1))
        """
        B, P, L = x.shape

        nan_counts = torch.isnan(x).sum(-1, keepdim=True)
        nan_counts = torch.cumsum(nan_counts, dim=1)

        counts = (
            torch.arange(1, P + 1, device=x.device).view(1, P, 1).repeat(B, 1, 1) * L
        )
        counts = counts - nan_counts


        cumsum_x = torch.cumsum(x.nansum(dim=-1, keepdim=True), dim=1)

        mean = cumsum_x / counts

        cumsum_x2 = torch.cumsum((x**2).nansum(dim=-1, keepdim=True), dim=1)

        var = (cumsum_x2 - 2 * mean * cumsum_x + counts * mean**2) / counts
        std = torch.sqrt(var + 1e-5)

        return mean, std

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
        dropout=0,
    ):
        super().__init__()

        self.patch_len = patch_len
        self.revin = CausalRevIN()

        self.proj_embedding = ResidualBlock(
            in_dim=2*patch_len, hid_dim=4 * patch_len, out_dim=d_model, dropout=dropout
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

        x = self.revin(x, mode="norm")
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
                    masked_batch[mask] = torch.nan
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
                 win_size = 128,
                 dim_embedding = 64,

                 feats = 1,
                 batch_size = 256,
                 ):
        super().__init__()

        self.__anomaly_score = None

        self.cuda = True
        self.device = get_gpu(self.cuda)

        self.win_size = win_size
        self.dim_embedding = dim_embedding
        self.batch_size = batch_size
        self.feats = feats

        self.model = Encoder(
            patch_len=32,
            d_model=512,
            n_heads=4,
            n_layers_encoder=6,
            dropout=0.0,
        ).to(self.device)

        print("Loading pre-trained JEPA encoder...")
        ckpt = torch.load("ckpts/jepa-epoch=06---step-step=175000.ckpt", weights_only=False)
        encoder_state_dict = {k.replace("jepa.encoder.", ""): v for k, v in ckpt["state_dict"].items() if k.startswith("jepa.encoder.")}
        self.model.load_state_dict(encoder_state_dict, strict=True)
        print("Pre-trained JEPA encoder loaded.")

    def fit(self, data):

        tsTrain = data
        if len(tsTrain) < self.win_size:
            self.win_size = 256
        if len(tsTrain) < self.win_size:
            self.win_size = 128

        train_loader = DataLoader(
            dataset=ReconstructDataset(tsTrain, window_size=self.win_size),
            batch_size=self.batch_size,
            shuffle=True
        )
        print("Extracting bank embeddings...")
        bank_embedding, pca = get_bank_embedding(self.model, train_loader, num_cores=500, dim_embedding=self.dim_embedding, device=self.device)
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

        self.model.eval()
        scores = compute_anomaly_score(self.model, test_loader, self.bank_embedding, self.pca, top_k=3, device=self.device)

        self.__anomaly_score = scores

        if self.__anomaly_score.shape[0] < len(data):
            self.__anomaly_score = np.array([self.__anomaly_score[0]]*math.ceil((self.win_size-1)/2) + 
                        list(self.__anomaly_score) + [self.__anomaly_score[-1]]*((self.win_size-1)//2))
        
        return self.__anomaly_score

    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score

    def param_statistic(self, save_file):
        pass
