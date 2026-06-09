import math

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from .utils.slidingWindows import find_length_rank

Unsupervise_AD_Pool = ['FFT', 'SR', 'NORMA', 'Series2Graph', 'Sub_IForest', 'IForest', 'LOF', 'Sub_LOF', 'POLY', 'MatrixProfile', 'Sub_PCA', 'PCA', 'HBOS',
                        'Sub_HBOS', 'KNN', 'Sub_KNN','KMeansAD', 'KMeansAD_U', 'KShapeAD', 'COPOD', 'CBLOF', 'COF', 'EIF', 'RobustPCA', 'MMPAD', 'Lag_Llama', 'TimesFM', 'Chronos', 'MOMENT_ZS', 'TSPulse_ZS', 'Time_RCD']
Semisupervise_AD_Pool = ['Left_STAMPi', 'SAND', 'MCD', 'Sub_MCD', 'OCSVM', 'Sub_OCSVM', 'AutoEncoder', 'CNN', 'LSTMAD', 'TranAD', 'USAD', 'OmniAnomaly', 'PatchTST',
                        'AnomalyTransformer', 'TimesNet', 'FITS', 'Donut', 'OFA', 'MOMENT_FT', 'M2N2', 'TSPulse_FT', 'xLSTMAD', 'CHARM', 'patchtrad', 'patchtradmv', 'CRIB', 'IMAD', 'robust', 'PaAno', 'JEPAno', 'MTAD']

def run_Unsupervise_AD(model_name, all_datas, clf, **kwargs):
    try:
        function_name = f'run_{model_name}'
        function_to_call = globals()[function_name]
        results, _ = function_to_call(all_datas, clf, **kwargs)
        return results, _
    except KeyError as e:
        error_message = f"Model function '{function_name}' is not defined."
        print(error_message)
        return error_message
    except Exception as e:
        error_message = f"An error occurred while running the model '{function_name}': {str(e)}"
        print(error_message)
        return error_message


def run_Semisupervise_AD(model_name, data_train, all_datas, clf, **kwargs):
    try:
        function_name = f'run_{model_name}'
        function_to_call = globals()[function_name]
        results, clf = function_to_call(data_train, all_datas, clf, **kwargs)
        return results, clf
    except KeyError as e:
        error_message = f"Model function '{function_name}' is not defined. {e}"
        print(error_message)
        return error_message
    except Exception as e:
        error_message = f"An error occurred while running the model '{function_name}': {str(e)}"
        print(error_message)
        return error_message

def run_MatrixProfile(data, clf=None, periodicity=1, n_jobs=1):
    from .models.MatrixProfile import MatrixProfile
    try:
        slidingWindow = find_length_rank(data["data_imputed"], rank=periodicity)
    except Exception as e:
        print(f"Error occurred while finding sliding window: {e}")
        return None, None
    clf = MatrixProfile(window=slidingWindow)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel(), None


def run_TranAD(data_train, all_datas, clf, win_size=10, lr=1e-3):
    from .models.TranAD import TranAD
    if clf is None:
        clf = TranAD(win_size=win_size, feats=all_datas["data"].shape[1], lr=lr)
        clf.fit(data_train)
    score = clf.decision_function(all_datas)
    return score.ravel(), clf

def run_patchtrad(data_train, all_datas, clf, win_size=10, lr=1e-3, patch_len=8, stride=6, d_model=128, n_heads=4, n_layers=3, d_ff=256):
    from .models.PatchTrAD import PatchTrAD
    if clf is None:
        clf = PatchTrAD(win_size=win_size, feats=all_datas["data"].shape[1], lr=lr, patch_len=patch_len, stride=stride, d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff)
        clf.fit(data_train)
    score = clf.decision_function(all_datas)
    return score.ravel(), clf

def run_MTAD(data_train, all_datas, clf, win_size=61):
    from .models.MTAD import MTAD_Detector
    if clf is None:
        clf = MTAD_Detector(win_size=win_size, feats=all_datas["data"].shape[1])
        clf.fit(data_train)
    score = clf.decision_function(all_datas)
    return score.ravel(), clf

def run_patchtradmv(data_train, all_datas, clf, win_size=10, lr=1e-3, patch_len=8, stride=6, d_model=128, n_heads=4, n_layers=3, d_ff=256):
    from .models.PatchTrAD_MV import PatchTrAD
    if clf is None:
        clf = PatchTrAD(win_size=win_size, feats=all_datas["data"].shape[1], lr=lr, patch_len=patch_len, stride=stride, d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff)
        clf.fit(data_train)
    score = clf.decision_function(all_datas)
    return score.ravel(), clf

def run_JEPAno(data_train, all_datas, clf, win_size=10, dim_embedding=64):
    from .models.JEPAno import JEPAno
    if clf is None:
        clf = JEPAno(win_size=win_size, dim_embedding=dim_embedding, feats=all_datas["data"].shape[1])
        clf.fit(data_train)
    score = clf.decision_function(all_datas)
    return score.ravel(), clf

def run_CRIB(data_train, all_datas, clf, d_model=32, seq_len=25, patch_len=8):
    from .models.CRIB import CRIB_Detector
    if clf is None:
        clf = CRIB_Detector(d_model=d_model, seq_len=seq_len, patch_len=patch_len)
        clf.fit(data_train)
    score = clf.decision_function(all_datas)
    return score.ravel(), clf

def run_robust(data_train, all_datas, clf, d_model=32, seq_len=25, patch_len=8):
    from .models.robust import CRIB_Detector
    if clf is None:
        clf = CRIB_Detector(d_model=d_model, seq_len=seq_len, patch_len=patch_len)
        clf.fit(data_train)
    score = clf.decision_function(all_datas)
    return score.ravel(), clf

def run_PCA(data, n_jobs=1):
    from .models.PCA import PCA
    clf = PCA()
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel(), None

def run_IMAD(data_train, all_datas, clf, mid_dim=128):
    from .models.IMAD import IMAD_Detector
    if clf is None:
        clf = IMAD_Detector(mid_dim=mid_dim)
        clf.fit(data_train)
    score = clf.decision_function(all_datas)
    return score.ravel(), clf

def run_PaAno(data_train, all_datas, clf, patch_size=96, num_iters=100, lr=1e-4, batch_size=512, validation_size=0.05):
    from .models.PaAno import PaAnoDetector
    if clf is None:
        clf = PaAnoDetector(patch_size=patch_size, num_iters=num_iters, lr=lr, feats=all_datas["data"].shape[1], batch_size=batch_size, validation_size=validation_size)
        clf.fit(data_train)
    score = clf.decision_function(all_datas)
    return score.ravel(), clf

def run_AnomalyTransformer(data_train, all_datas, clf, win_size=100, lr=1e-4, batch_size=128):
    from .models.AnomalyTransformer import AnomalyTransformer
    if clf is None:
        clf = AnomalyTransformer(win_size=win_size, input_c=all_datas["data"].shape[1], lr=lr, batch_size=batch_size)
        clf.fit(data_train)
    score = clf.decision_function(all_datas)
    return score.ravel(), clf

def run_PatchTST(data_train, all_datas, clf, win_size=100, lr=1e-4, batch_size=128):
    from .models.PatchTST import PatchTST
    if clf is None:
        clf = PatchTST(win_size=win_size, input_c=all_datas["data"].shape[1], lr=lr, batch_size=batch_size)
        clf.fit(data_train)
    score = clf.decision_function(all_datas)
    return score.ravel(), clf

def run_USAD(data_train, all_datas, clf, win_size=5, lr=1e-4):
    from .models.USAD import USAD
    if clf is None:
        clf = USAD(win_size=win_size, feats=all_datas["data"].shape[1], lr=lr)
        clf.fit(data_train)
    score = clf.decision_function(all_datas)
    return score.ravel(), clf