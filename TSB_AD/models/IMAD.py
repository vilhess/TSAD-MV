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

def get_missing_data(data, missing_rate, mechanism='mcar'):


    if mechanism == 'mcar':
        mask = np.random.rand(*data.shape) < missing_rate # True for missing values, false for others
        mask = torch.from_numpy(mask)
    else:
        raise NotImplementedError(f"Missing data mechanism {mechanism} not implemented")

    data[mask] = 0.0

    return data, 1 - 1 * mask

# Adapted from https://github.com/gpeyre/SinkhornAutoDiff
class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """
    def __init__(self, eps=0.1, max_iter=2000, thresh=1e-3, reduction='none', device='cpu'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.thresh = thresh
        self.reduction = reduction
        self.device = device
        print('=============== SinkHorn ===============')
        print(f'========= epsilon:{self.eps}')
        print(f'========= max iteration:{self.max_iter}')
        print(f'========= stop threshold:{self.thresh}')
        print('=============== SinkHorn ===============')

    def forward(self, x, y, normalized=False):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y, normalized=normalized)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).squeeze()
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).squeeze()

        u = torch.zeros_like(mu).to(self.device)
        v = torch.zeros_like(nu).to(self.device)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = self.thresh
        # thresh = 1e-3

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu+1e-8).to(self.device) - torch.logsumexp(self.M(C, u, v).to(self.device), dim=-1)) + u
            v = self.eps * (torch.log(nu+1e-8).to(self.device) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1).to(self.device), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                # print(f'error:{err.item()}')
                break
        # if actual_nits == self.max_iter:
        #     print('meeting max iteration.')            
        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        # return cost, pi, C
        return cost

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2, normalized=False):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        if normalized:
            C = C / torch.norm(C)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1

class Imputer(nn.Module):

    def __init__(self, input_dim):
        super(Imputer, self).__init__()

        self.input_dim = input_dim
        self.impute = nn.Sequential(
                nn.Linear(self.input_dim, 512, bias=False),
                nn.LeakyReLU(inplace=True),

                nn.Linear(512, 512, bias=False),
                nn.LeakyReLU(inplace=True),

                nn.Linear(512, self.input_dim, bias=False),
            )


    def forward(self, x):

        return self.impute(x)


class Projector(nn.Module):

    def __init__(self, input_dim, mid_dim):
        super(Projector, self).__init__()
        self.input_dim = input_dim
        self.mid_dim = mid_dim

        self.project = nn.Sequential(
                nn.Linear(self.input_dim, 512, bias=False),
                nn.LeakyReLU(inplace=True),

                nn.Linear(512, 256, bias=False),
                nn.LeakyReLU(inplace=True),

                nn.Linear(256, self.mid_dim, bias=False),
            )

    def forward(self, x):

        return self.project(x)


class Recover(nn.Module):

    def __init__(self, mid_dim, recover_dim) -> None:
        super(Recover, self).__init__()
        self.mid_dim = mid_dim
        self.recover_dim = recover_dim

        self.decoder = nn.Sequential(
                nn.Linear(self.mid_dim, 200, bias=False),
                nn.LeakyReLU(inplace=True),

                nn.Linear(200, self.recover_dim, bias=False)
            )

    def forward(self, x):
        
        return self.decoder(x)


class MVNet(nn.Module):

    def __init__(self, input_dim, mid_dim):
        super(MVNet, self).__init__()

        self.imputer = Imputer(input_dim=input_dim)
        self.projector = Projector(input_dim=input_dim, mid_dim=mid_dim)
        self.recover = Recover(mid_dim=mid_dim, recover_dim=input_dim)

    def forward(self, x, negative=False, missing_rate=0.0, mechanism=None):

        if not negative:
            imputed_data = self.imputer(x)
            mid_repre = self.projector(imputed_data)
            recover_data = self.recover(mid_repre)
            return imputed_data, mid_repre, recover_data
        else:
            recover_data = self.recover(x)
            
            missing_data, masks = get_missing_data(recover_data.cpu().data, missing_rate, mechanism)
            missing_data = missing_data.to('cuda')
            imputed_data = self.imputer(missing_data)
            mid_repre = self.projector(imputed_data)
            return imputed_data, mid_repre, missing_data, masks     
        

class IMAD_Detector(BaseDetector):
    def __init__(self,
                 seq_len = 25,
                 lr = 1e-3,
                 mid_dim = 128,
                 beta = 1.0,
                 lambda_recon = 1.0,
                 alpha = 1.0,
                 entropy_reg_coe = 0.01, 
                 stop_threshold = 1e-4,
                 mu = 0.0,
                 std = 1,

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

        if mid_dim == 256:
            r_min = 8.45
            r_max = 16.90
        elif mid_dim == 128:
            r_min = 6.10
            r_max = 12.20
        else:
            raise ValueError(f"Unsupported mid_dim {mid_dim}")

        self.win_size = seq_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.feats = feats
        self.validation_size = validation_size
        self.lr = lr
        self.beta = beta
        self.lambda_recon = lambda_recon
        self.alpha = alpha
        self.r_min = r_min
        self.r_max = r_max
        self.mu = mu
        self.std = std

        self.max_train_mask_proba = 0.3
        self.entropy_reg_coe = entropy_reg_coe
        self.stop_threshold = stop_threshold

        self.model = MVNet(input_dim=self.feats*self.win_size, mid_dim=mid_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.early_stopping = EarlyStoppingTorch(None, patience=patience)

    def fit(self, data):

        tsTrain = data[:int((1-self.validation_size)*len(data))]
        tsValid = data[int((1-self.validation_size)*len(data)):]

        sinkhorn_loss_fn = SinkhornDistance(eps=self.entropy_reg_coe, max_iter=int(1e2), thresh=self.stop_threshold, device=self.device)

        train_loader = DataLoader(
            dataset=ReconstructDataset(tsTrain, window_size=self.win_size, normalize=True),
            batch_size=self.batch_size,
            shuffle=True
        )
        
        valid_loader = DataLoader(
            dataset=ReconstructDataset(tsValid, window_size=self.win_size, normalize=True),
            batch_size=self.batch_size,
            shuffle=False
        )
        
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
                if bs == 1:
                    x = x.repeat(2, 1, 1)
                    bs = 2

                mask_proba = self.max_train_mask_proba * np.random.rand()

                mask = torch.bernoulli(torch.ones_like(x)*mask_proba).to(x.device)
                x = x.masked_fill(mask.bool(), 0)
                x = x.flatten(start_dim=1)
                mask = mask.flatten(start_dim=1)

                imputed_data, mid_repre, recover_data = self.model(x)
                imputed_loss = torch.sum( torch.mul((imputed_data - x), 1-mask)**2, dim=tuple(range(1, x.dim()))) * self.beta
                loss = torch.mean(imputed_loss)

                targets = target_distribution_sampling(bs, mid_repre[0].shape, r_min=self.r_min, r_max=self.r_max, mu=self.mu, std=self.std)
                targets = targets.to(self.device)
                dist_loss = sinkhorn_loss_fn(mid_repre, targets)
                loss += dist_loss

                recon_loss = torch.mean(torch.sum(torch.mul(
                    recover_data - x, 1-mask)**2, dim=tuple(range(1, x.dim()
                ))))*self.lambda_recon
                loss += recon_loss

                negative_samples = target_distribution_sampling(bs, mid_repre[0].shape, mu=self.mu, std=self.std, r_min=self.r_min, r_max=self.r_max).to(self.device)
                imputed_neg, mid_repre_neg, missing_data_neg, masks_neg = self.model(negative_samples, negative=True, missing_rate=mask_proba, mechanism='mcar')
                mask_neg = masks_neg.to(self.device)
                negative_loss = torch.mean(torch.sum(torch.mul(
                    imputed_neg - missing_data_neg, mask_neg)**2, dim=tuple(range(1, x.dim())
                )))*self.beta
                loss += negative_loss

                neg_dist_loss = torch.mean(torch.sum((mid_repre_neg - negative_samples)**2, dim=tuple(range(1, mid_repre_neg.dim()))))*self.alpha
                loss += neg_dist_loss

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

                        current_mask_proba = self.max_train_mask_proba * np.random.rand()
                        mask = torch.bernoulli(torch.ones_like(x)*current_mask_proba).to(x.device)
                        x = x.masked_fill(mask.bool(), 0)
                        
                        x = x.flatten(start_dim=1)
                        _, mid_repre, _ = self.model(x)
                        score = torch.sqrt(torch.sum(mid_repre**2, dim=tuple(range(1, mid_repre.dim()))))
                        loss = torch.mean(score)

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

                x = x.flatten(start_dim=1)
                _, mid_repre, _ = self.model(x)
                loss = torch.sqrt(torch.sum(mid_repre**2, dim=tuple(range(1, mid_repre.dim()))))
                
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

def target_distribution_sampling(size, sample_dim, r_max=None, r_min=0.0, mu=0, std=1):

    '''
    :params
    size:
    sample_dim: the dimension of the sample from the restricted distribution
    mu: the mean of normal distribution
    std: the standard devariate of normal distribution

    '''

    Sampler = randn
    targets = None
    
    while size > 0:

        sample = Sampler(sample_dim, mean=mu, std=std)
        sample_norm = torch.sqrt(torch.sum(sample ** 2))

        if r_min < sample_norm < r_max:
            if targets is None:
                targets = sample.unsqueeze(0)
            else:
                targets = torch.cat((targets, sample.unsqueeze(0)))
            size -= 1
    return targets


def randn(sample_dim, mean=0.0, std=1.0):

    '''
    N(0, 1) Gaussian
    '''
    return torch.distributions.Normal(loc=mean, scale=std).sample(sample_dim).squeeze(-1)