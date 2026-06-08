import pandas as pd
import numpy as np
import torch
import argparse, time, os

from TSB_AD.missing_modules.masker import get_mask

from benchmark_exp.configs import MISSING_RATES, MECHANISMS, IMPUTERS, DEFAULT_PATHS
from benchmark_exp.utils import set_seed, print_cuda_info
from benchmark_exp.data_loader import load_file, train_split

# seeding
seed = 2024
set_seed(seed)

print_cuda_info()

if __name__ == '__main__':

    Start_T = time.time()
    parser = argparse.ArgumentParser(description='Pre-compute imputations')
    parser.add_argument('--dataset_dir', type=str, default=DEFAULT_PATHS['dataset_dir'])
    parser.add_argument('--file_list', type=str, default=DEFAULT_PATHS['file_list'])
    parser.add_argument('--save_imputation_dir', type=str, default=DEFAULT_PATHS['load_imputation_dir'])
    args = parser.parse_args()

    file_list = pd.read_csv(args.file_list)['file_name'].values

    trainable_imputers = {name: imp for name, imp in IMPUTERS.items() if getattr(imp, 'trainable', False)}

    for i, filename in enumerate(file_list):
        print(f"Processing file {i+1}/{len(file_list)}: {filename}")

        file_path = os.path.join(args.dataset_dir, filename)
        data, _, _ = load_file(file_path)
        data_train = train_split(data, filename)

        for imputer in trainable_imputers.values():
            imputer.fit(data_train, prc_val=0.25)

            for mechanism in MECHANISMS:
                for missing_rate in MISSING_RATES:

                    mask = get_mask(torch.from_numpy(data), seed=seed*i*(1+10*missing_rate), missing_rate=missing_rate, strategy=mechanism).numpy()

                    data_missing = data.copy()
                    data_missing[mask == 1] = np.nan

                    data_imputed = imputer.impute(data_missing, mask)
                    print(f"True missing rate: {missing_rate}, Missing rate observed %: {mask.sum()/ mask.size:.2%}")

                    save_dir = f'{args.save_imputation_dir}/{mechanism}/{imputer.imputer_name}/{missing_rate}'
                    os.makedirs(save_dir, exist_ok=True)
                    np.save(f'{save_dir}/{filename}_data_imputed.npy', data_imputed)
