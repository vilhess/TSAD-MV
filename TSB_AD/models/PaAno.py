from __future__ import division
from __future__ import print_function

import numpy as np
import math
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from torch.utils.data import DataLoader
from sklearn.cluster import MiniBatchKMeans
import copy
import tqdm

from .base import BaseDetector
from ..utils.dataset import ReconstructCombinedDataset
from ..utils.torch_utility import get_gpu

class RevIN1d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-5, min_sigma: float = 1e-5, affine: bool = False):
        super().__init__()
        self.eps = eps
        self.min_sigma = min_sigma
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1))
            self.bias   = nn.Parameter(torch.zeros(1, num_channels, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self._mu = None
        self._sigma = None

    @torch.no_grad()
    def _stats(self, x):
        mu = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        sigma = (var + self.eps).sqrt().clamp_min(self.min_sigma)
        return mu, sigma

    def norm(self, x):
        self._mu, self._sigma = self._stats(x)
        
        x_hat = (x - self._mu) / self._sigma
        
        if self.affine:
            x_hat = x_hat * self.weight + self.bias
        
        return x_hat

    def denorm(self, x_hat):
        mu, sigma = self._mu, self._sigma
        if mu is None or sigma is None:
            raise RuntimeError("Call norm() before denorm().")
        if self.affine:
            w = self.weight if self.weight is not None else 1.0
            b = self.bias if self.bias is not None else 0.0
            x_hat = (x_hat - b) / (w + self.eps)
        return x_hat * sigma + mu

class PatchEncoder(nn.Module): #Simple 1D CNN with RevIN
    def __init__(self, in_channels=1, projection_dim=256, layers=[128, 256, 128, 64],
                 kss=[7, 5, 3, 3],
                 use_revin: bool = True,       
                 revin_affine: bool = False,   
                 revin_eps: float = 1e-5,      
                 revin_min_sigma: float = 1e-5 
                 ):
        super(PatchEncoder, self).__init__()
        self.layers = layers
        self.kss = kss
        self.projection_dim = projection_dim

        # RevIN 
        self.revin = None
        if use_revin:
            self.revin = RevIN1d(num_channels=in_channels,
                                 eps=revin_eps,
                                 min_sigma=revin_min_sigma,
                                 affine=revin_affine)

        #  Conv blocks 
        self.convblocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(layers[i - 1] if i > 0 else in_channels, self.layers[i],
                          kernel_size=self.kss[i], stride=1, padding=self.kss[i] // 2, bias=False),
                nn.BatchNorm1d(self.layers[i]),
                nn.ReLU(inplace=True)
            ) for i in range(len(self.layers))
        ])

        # Heads 
        self.fc_embedding = nn.AdaptiveAvgPool1d(output_size=1)
        self.gap = nn.AdaptiveAvgPool1d(output_size=1)
        self.projection_head = nn.Sequential(
            nn.Linear(self.layers[-1], self.projection_dim),
            nn.ReLU(),
            nn.Linear(self.projection_dim, self.projection_dim)
        )
        self.classification_head = nn.Linear(self.layers[-1]*2, 1)

    def forward(self, x, return_embedding=False, return_projection=False):
        if self.revin is not None:
            x = self.revin.norm(x)  

        for block in self.convblocks:
            x = block(x)

        h = self.fc_embedding(x).flatten(start_dim=1)  # (N, D)

        if return_embedding:
            return h
        if return_projection:
            return self.projection_head(h)

        raise ValueError("The forward method is not designed to handle classification directly.")

    def embedding(self, x):
        return self.forward(x, return_embedding=True)

    def projection(self, h):
        return self.projection_head(h)

class _tsdataset(torch.utils.data.Dataset):
    def __init__(self, data, indices=None):
        # indice means relative order among patches 
        self.data = torch.from_numpy(np.array(data)).float()
        if indices is not None:
            self.indices = torch.from_numpy(np.array(indices)).long().unsqueeze(1)
        else:
            self.indices = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
            # x: (L,) or (L, C) or (C, L)
        if x.ndim == 1:          # (L,) -> (1, L)
            x = x.unsqueeze(0).contiguous()
       
        if self.indices is not None:
            return x, self.indices[idx]
        return x, torch.tensor([idx])

def preprocess_to_patches(data, patch_size, stride):
    patches = []
    for i in range(0, len(data) - patch_size + 1, stride):
        patch = data[i:i + patch_size]
        patches.append(patch)
    
    patches_array = np.array(patches)                      # (N, L) or (N, L, C)
    t = torch.tensor(patches_array, dtype=torch.float32)
    if t.ndim == 2:                   # (N, L) -> (N, 1, L)
        t = t.unsqueeze(1).contiguous()
    elif t.ndim == 3:                 # (N, L, C) -> (N, C, L)
        t = t.permute(0, 2, 1).contiguous()

    return t  

class PatchCreator:
    def __init__(self, L, s, random_seed=None):
        self.L = L 
        self.s = s  
        if random_seed is not None:
            torch.manual_seed(random_seed)  

    def create_patches(self, data):
        if not isinstance(data, (list, np.ndarray)):
            raise ValueError("Data must be a list or numpy array.")
        if len(data) < self.L:
            raise ValueError(f"Data length {len(data)} is less than patch size {self.L}.")
    
        num_patches = (len(data) - self.L) // self.s + 1
        patches = [data[i:i+self.L] for i in range(0, len(data) - self.L + 1, self.s)]
        indices = [i for i in range(0, len(data) - self.L + 1, self.s)]
        return patches, indices

    def create_dataloader(self, data, batch_size=512):
        patches, indices = self.create_patches(data)
        patches = [p.T for p in patches] if patches[0].ndim == 2 else patches
        loader = DataLoader(_tsdataset(patches, indices=indices), batch_size=batch_size, shuffle=True)
        return loader

def train_model(model, train_loader, train_patches, device, num_iter=200, pretext_step=64,
                lr=1e-4, see_loss=None):

    # fixed hyperparams in PaAno
    radius = 2
    lambda_weight = 1
    temperature = 1.0
    num_rand_patches = 5
    initial_lr = lr
    final_lr = lr / 10

    def cosine_annealed_lr(iteration):
        t = min(iteration, num_iter)
        cosine_factor = 0.5 * (1 + math.cos(math.pi * t / num_iter))
        return final_lr + (initial_lr - final_lr) * cosine_factor

    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-4)
    pos_weight = torch.tensor([1.0]).to(device)
    criterion_pretext = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')

    iteration_count = 0
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())

    print("    [Training Info]")
    pbar = tqdm.tqdm(total=num_iter, desc="    >> Training", ncols=80)

    _offsets = torch.tensor([*range(-radius, 0), *range(1, radius + 1)], dtype=torch.long)

    while iteration_count < num_iter:
        for batch_data, batch_indexes in train_loader:
            if iteration_count >= num_iter:
                break

            iteration_count += 1

            # Update LR
            lr = cosine_annealed_lr(iteration_count)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            batch_data = batch_data.to(device, non_blocking=True)
            batch_indexes = batch_indexes.squeeze()  # (M,)
            anchors = batch_data
            M = batch_data.shape[0]
            mu = 10
            total_len = len(train_patches)

            # positives 
            _cand = batch_indexes.view(-1, 1) + _offsets.view(1, -1)      # (M, 2r)
            _valid = (_cand >= 0) & (_cand < total_len)
            _noise = torch.rand_like(_cand.float())
            _score = torch.where(_valid, _noise, torch.full_like(_noise, -1.0))
            _choice = _score.argmax(dim=1)                                # (M,)
            _pos_idx = _cand.gather(1, _choice.view(-1, 1)).squeeze(1)    # (M,)
            _none_valid = _valid.sum(dim=1) == 0
            if _none_valid.any():
                _pos_idx[_none_valid] = batch_indexes[_none_valid]
            positives = torch.stack([train_patches[i] for i in _pos_idx.tolist()], dim=0).to(device, non_blocking=True)

            if iteration_count < (num_iter / 5) :
                current_lambda_pretext = lambda_weight * (1 - (iteration_count / (num_iter / 5)))
            else:
                current_lambda_pretext = 0.0

            if current_lambda_pretext > 0.0:
                # pretext_patches 
                pretext_patches = []
                pretext_valid_mask = []

                _tgt = batch_indexes - pretext_step
                _pre_mask = (_tgt >= 0) & (_tgt < total_len)
                _tgt_clamped = _tgt.clamp(0, total_len - 1)

                for i in range(M):
                    if _pre_mask[i]:
                        pretext_patches.append(train_patches[_tgt_clamped[i].item()].unsqueeze(0))
                        pretext_valid_mask.append(True)
                    else:
                        pretext_patches.append(torch.zeros_like(train_patches[0].unsqueeze(0)))
                        pretext_valid_mask.append(False)

                pretext_patches = torch.cat(pretext_patches, dim=0).to(device, non_blocking=True)
                pretext_valid_mask = torch.tensor(pretext_valid_mask, dtype=torch.bool, device=device)

                # anchors + positives + pretext
                all_patches = torch.cat([anchors, positives, pretext_patches], dim=0)
                all_embeddings = model.embedding(all_patches)

                h_anchors = all_embeddings[:M]
                h_pos     = all_embeddings[M:2*M]
                h_pretext = all_embeddings[2*M:3*M]

            else:
                pretext_patches    = None
                pretext_valid_mask = None

                # anchors + positives
                all_patches = torch.cat([anchors, positives], dim=0)
                all_embeddings = model.embedding(all_patches)

                h_anchors = all_embeddings[:M]
                h_pos     = all_embeddings[M:2*M]

            # triplet
            z_anchor = model.projection(h_anchors)
            z_pos    = model.projection(h_pos)

            z_anchor = F.normalize(z_anchor, dim=1)
            z_pos    = F.normalize(z_pos, dim=1)

            _sim_ap  = (z_anchor @ z_pos.T) / temperature         # (M, M)
            pos_sims = _sim_ap.diag()                             # (M,)

            _sim_ap_f = _sim_ap.clone()
            _sim_ap_f.diagonal().fill_(+float('inf')) 
            neg_dists = 1 - _sim_ap_f
            hard_neg_dists, _ = torch.max(neg_dists, dim=1)

            pos_dists = 1 - pos_sims
            triplet_loss = F.relu(pos_dists - hard_neg_dists + 0.1).mean() / mu
        

            # Pretext Task 
            if current_lambda_pretext > 0.0:
                h_pre = h_pretext[pretext_valid_mask]
                h_anchor_pre = h_anchors[pretext_valid_mask]
                h_concat_pre = torch.cat([h_anchor_pre, h_pre], dim=1)

                all_indices = torch.arange(M, device=device)
                anchor_indices = all_indices.repeat_interleave(num_rand_patches)
                rand_offsets = torch.randint(1, M, (M * num_rand_patches,), device=device)
                unadj_indices = (anchor_indices + rand_offsets) % M

                h_unadj = h_anchors[unadj_indices]
                h_anchor_unadj = h_anchors.repeat_interleave(num_rand_patches, dim=0)
                h_concat_unadj = torch.cat([h_anchor_unadj, h_unadj], dim=1)

                all_pretext_features = torch.cat([h_concat_pre, h_concat_unadj], dim=0)
                all_pretext_labels = torch.cat([
                    torch.ones(h_concat_pre.size(0), device=device),
                    torch.zeros(h_concat_unadj.size(0), device=device)
                ])

                pretext_outputs = model.classification_head(all_pretext_features).squeeze(1)
                pretext_loss_all = criterion_pretext(pretext_outputs, all_pretext_labels)

                loss_pre = pretext_loss_all[:h_concat_pre.size(0)].mean()
                loss_unadj = pretext_loss_all[h_concat_pre.size(0):].mean()
                pretext_loss = loss_pre + loss_unadj
            else:
                pretext_loss = torch.tensor(0.0, device=device)

            final_loss = triplet_loss + current_lambda_pretext * pretext_loss

            optimizer.zero_grad(set_to_none=True)
            final_loss.backward()
            optimizer.step()

            pbar.update(1)
          
            if final_loss.item() < best_loss:
                best_loss = final_loss.item()
                best_model_wts = copy.deepcopy(model.state_dict())
    print(f"    >> Best Loss: {best_loss:.4f}")
    pbar.close()
    return best_model_wts

@torch.no_grad()
def create_memory_bank(model, data_loader, device, num_cores=None):  
    model.eval()
    embeddings = []
    indices = []
    
    for data, batch_indices in data_loader:
        data = data.to(device)
        h = model.embedding(data)
        embeddings.append(h.detach().cpu().float())  
        indices.append(batch_indices.detach().cpu())

    embeddings_tensor = torch.cat(embeddings, dim=0)
    indices_tensor    = torch.cat(indices, dim=0)
    num_samples       = embeddings_tensor.size(0)

 
    if num_cores is None:
        return embeddings_tensor, indices_tensor

  
    if isinstance(num_cores, float):
        k = int(round(num_cores * num_samples))    
    else:
        k = int(num_cores)

    min_cores_eff = min(500, max(1, num_samples - 1)) 
    num_cores = max(min_cores_eff, min(k, num_samples - 1))

    if num_cores >= num_samples:
        return embeddings_tensor, indices_tensor

  
    flattened = embeddings_tensor.view(num_samples, -1)
    flattened = F.normalize(flattened, p=2, dim=1)

 
    mbk = MiniBatchKMeans(
        n_clusters=num_cores,
        init='k-means++',
        random_state=42,
        batch_size=max(8192, num_cores),   
        max_iter=50,                       
        n_init=1,                          
        reassignment_ratio=0.01
    )
    mbk.fit(flattened.numpy())


    centers = torch.tensor(mbk.cluster_centers_, dtype=flattened.dtype)  
    distances = torch.cdist(flattened, centers, p=2)   
    core_indices = torch.argmin(distances, dim=0)      

    embeddings_tensor = embeddings_tensor[core_indices]
    indices_tensor    = indices_tensor[core_indices]

    return embeddings_tensor, indices_tensor

@torch.inference_mode()
def calculate_anomaly_scores(model, data_loader, memory_bank, device, top_k=3):
    
    model.eval()
    all_scores = []
    memory_bank = F.normalize(memory_bank.to(device, dtype=torch.float32), dim=1, eps=1e-12)


    for data, _ in data_loader:
        data = data.to(device, non_blocking=True, dtype=torch.float32).permute(0, 2, 1)
        feats = model.embedding(data)  # (B, D)
        feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

        feats = F.normalize(feats, dim=1, eps=1e-12)
        feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

        # Cosine similarity & distance
        sims = feats @ memory_bank.T                    # (B, M)
        sims = torch.nan_to_num(sims, nan=-1.0, posinf=1.0, neginf=-1.0)
        topk_sim, _ = torch.topk(sims, k=top_k, dim=1, largest=True)
        dists = 1.0 - topk_sim
        scores = dists.mean(dim=1)

        scores = torch.nan_to_num(scores, nan=1.0, posinf=1.0, neginf=0.0)
        all_scores.extend(scores.cpu().tolist())

    return all_scores

class PaAnoDetector(BaseDetector):
    def __init__(self,
                 patch_size=96,
                 num_iters=100,
                 lr=1e-4,
                 batch_size = 512,

                 feats = 1,
                 validation_size=0.05
                 ):
        super().__init__()

        self.__anomaly_score = None

        self.cuda = True
        self.device = get_gpu(self.cuda)
        self.patch_size = patch_size
        self.lr = lr

        self.patch_size = patch_size
        self.batch_size = batch_size
        self.num_iters = num_iters
        self.feats = feats
        self.validation_size = validation_size

        self.model = PatchEncoder(in_channels=self.feats, use_revin=True).to(self.device)

    def fit(self, data):

        tsTrain = data[:int((1-self.validation_size)*len(data))]
        patch_creator = PatchCreator(L=self.patch_size, s=1, random_seed=42)

        train_loader = patch_creator.create_dataloader(tsTrain, batch_size=self.batch_size)
        train_patches = preprocess_to_patches(tsTrain, patch_size=self.patch_size, stride=1)

        state_dict = train_model(model=self.model, train_loader=train_loader, train_patches=train_patches, device=self.device,
                                 num_iter=self.num_iters, pretext_step=self.patch_size, lr=self.lr)
        self.model.load_state_dict(state_dict)

        print("    >> Creating Memory Bank...")
        self.memory_bank, _ = create_memory_bank(self.model, train_loader, self.device, num_cores=0.1)
        print(f"    >> Memory Bank Size: {self.memory_bank.size(0)}")

    def decision_function(self, all_datas):
        data = all_datas["data_imputed"]
        true_data = all_datas["data"] # last timestep per window never contains missing value 
    
        test_loader = DataLoader(
            dataset=ReconstructCombinedDataset(data, true_data, window_size=self.patch_size),
            batch_size=self.batch_size,
            shuffle=False,
        )

        self.model.eval()
        scores = calculate_anomaly_scores(self.model, test_loader, self.memory_bank, self.device, top_k=3)
        scores = np.array(scores)

        self.__anomaly_score = scores

        if self.__anomaly_score.shape[0] < len(data):
            self.__anomaly_score = np.array([self.__anomaly_score[0]]*math.ceil((self.patch_size-1)/2) + 
                        list(self.__anomaly_score) + [self.__anomaly_score[-1]]*((self.patch_size-1)//2))
        
        return self.__anomaly_score

    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score

    def param_statistic(self, save_file):
        pass
