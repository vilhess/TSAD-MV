import os
import pandas as pd
import numpy as np
from TSB_AD.utils.slidingWindows import find_length_rank

def load_file(file_path):
    df = pd.read_csv(file_path).dropna()
    data = df.iloc[:, 0:-1].values.astype(float)
    label = df['Label'].astype(int).to_numpy()
    slidingWindow = find_length_rank(data[:,0].reshape(-1, 1), rank=1)
    return data, label, slidingWindow

def train_split(data, filename):
    train_index = filename.split('.')[0].split('_')[-3]
    data_train = data[:int(train_index), :]
    return data_train

def get_dict_data(train_data, data, mask, imputer, missing_rate, saved_imputation_dir, filename, mechanism):
    data_missing = data.copy()
    data_missing[mask == 1] = np.nan
    print(f"True missing rate: {missing_rate}, Missing rate observed %: {mask.sum()/ mask.size:.2%}")

    if saved_imputation_dir is not None:
        imputed_path = f'{saved_imputation_dir}/{mechanism}/{imputer.imputer_name}/{missing_rate}/{filename}_data_imputed.npy'
        if os.path.exists(imputed_path):
            print(f"Loading imputed data from {imputed_path}")
            data_imputed = np.load(imputed_path)
        else:
            print(f"No imputed data found at {imputed_path}, performing imputation...")
            data_imputed = imputer.impute(data_missing, mask)

    assert np.allclose(data_imputed[mask == 0], data[mask == 0]), "Imputed values do not match original values where mask is 0"
    print(f"Imputation completed for {filename} with missing rate {missing_rate} using {imputer.imputer_name}.")

    
    all_datas = {
        "data": data,
        "data_missing": data_missing,
        "data_imputed": data_imputed, 
        "mask": mask
    }
    return all_datas
