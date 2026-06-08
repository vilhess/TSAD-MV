import numpy as np
from pygrinder import mcar
from pypots.imputation import SAITS

class SAITSConfig:
    def __init__(self, in_dim, ws=64, n_layers=2, d_model=256, n_heads=4, d_k=64, d_v=64, d_ffn=128, dropout=0.1, epochs=20):
        self.imputer_name = 'saits'
        self.ws = ws
        self.in_dim = in_dim
        self.epochs = epochs

        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.d_ffn = d_ffn
        self.dropout = dropout

        self.model = SAITS(n_steps=ws, n_features=in_dim, n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ffn=d_ffn, dropout=dropout, epochs=epochs)
        
    def fit(self, trainset, prc_val):

        x_sliced = []
        for i in range(0, len(trainset) - self.ws + 1):
            x_sliced.append(trainset[i:i+self.ws])

        data = np.array(x_sliced)
        print(f"Trainset shape: {data.shape}")

        x_missing = mcar(data, p=0.2) # introduce missingness
        n_train = int(len(x_sliced) * (1 - prc_val))
        train_X, val_X = x_missing[:n_train], x_missing[n_train:]
        trainset = {"X": train_X}
        valset = {"X": val_X, "X_ori": data[n_train:]}
        print(f"Start fitting SAITS model...")
        self.model.fit(trainset, valset)
        print(f"SAITS model training completed.")

    def naive_impute(self, window, mask):
        window = window.copy()  # avoid modifying the original window
        window = window[np.newaxis, :, :]  # add batch dimension
        mask = mask[np.newaxis, :, :]  # add batch dimension to mask
        window = {"X": window}
        imputed_window = self.model.impute(window)
        imputed_window[mask==0] = window["X"][mask==0]  # keep original values where mask is 0
        return imputed_window.squeeze(0)  # remove batch dimension
    
    def impute(self, window, mask):
        print("Simulation of online imputation with SAITS...")
        window[mask == 1] = np.nan
        for i in range(len(window)-self.ws+1):
            window_slice = window[i:i+self.ws]
            current_mask = np.isnan(window_slice)
            if np.any(current_mask):
                imputed_slice = self.naive_impute(window_slice, current_mask)
                window[i:i+self.ws][current_mask] = imputed_slice[current_mask]
        print("SAITS imputation completed.")
        return window