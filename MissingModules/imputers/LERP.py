import numpy as np
from pypots.imputation import Lerp

class LERPimputer:
    def __init__(self):
        self.model = Lerp()
        self.imputer_name = "LERP"
        
    def fit(self, trainset, prc_val):
        raise NotImplementedError("LERP does not require training. The fit method is not implemented.")

    def impute(self, window, mask):
        window[mask == 1] = np.nan

        window = window[np.newaxis, :, :]

        window = {"X": window}
        imputed_window = self.model.predict(window)
        output = imputed_window["imputation"]
        output = output.squeeze(0)
        
        return output