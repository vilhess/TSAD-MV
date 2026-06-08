class FillBy0imputer:
    def __init__(self):
        self.model = None  # FillBy0 does not require a model, but we keep this for consistency with other imputers.
        self.imputer_name = "FillBy0"
        
    def fit(self, trainset, prc_val):
        raise NotImplementedError("FillBy0 does not require training. The fit method is not implemented.")

    def impute(self, window, mask):
        window[mask == 1] = 0
        return window