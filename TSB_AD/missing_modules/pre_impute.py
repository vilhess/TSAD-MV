import pandas as pd
import numpy as np
import torch
import random, argparse, time, os, logging

from TSB_AD.missing_modules.masker import get_mask
from TSB_AD.missing_modules.imputers.SAITS import SAITSConfig

from benchmark_exp.configs import MISSING_RATES, MECHANISMS
from benchmark_exp.utils import set_seed, print_cuda_info
from benchmark_exp.data_loader import load_file, train_split

# seeding
seed = 2024
set_seed(seed)

print_cuda_info()

if __name__ == '__main__':

    Start_T = time.time()
    ## ArgumentParser
    parser = argparse.ArgumentParser(description='Generating Anomaly Score')
    parser.add_argument('--dataset_dir', type=str, default='/home/svilhes/Bureau/uad2/TSB-AD-U/')
    parser.add_argument('--file_lsit', type=str, default='Datasets/File_List/TSB-AD-U-Eva.csv')
    parser.add_argument('--save_imputation_dir', type=str, default='/home/svilhes/Bureau/uad2/TSB-AD-U-imputed/')
    parser.add_argument('--imputer', type=str, default='saits')
    args = parser.parse_args()

    file_list = pd.read_csv(args.file_lsit)['file_name'].values

    for i, filename in enumerate(file_list):
        print(f"Processing file {i+1}/{len(file_list)}: {filename}")

        file_path = os.path.join(args.dataset_dir, filename)
        data, label, slidingWindow = load_file(file_path)
        data_train = train_split(data, filename)

        clf = None

        for mechanism in MECHANISMS:
            imputer = SAITSConfig(in_dim=data.shape[1])
            imputer.fit(data_train, prc_val=0.25)
            for missing_rate in MISSING_RATES:

                mask = get_mask(torch.from_numpy(data), seed=seed*i*(1+10*missing_rate), missing_rate=missing_rate, strategy=mechanism).numpy()
                
                data_missing = data.copy()
                data_missing[mask == 1] = np.nan

                data_imputed = imputer.impute(data_missing, mask)
                print(f"True missing rate: {missing_rate}, Missing rate observed %: {mask.sum()/ mask.size:.2%}")
                os.makedirs(f'{args.save_imputation_dir}/{mechanism}/{args.imputer}/{missing_rate}', exist_ok=True)
                np.save(f'{args.save_imputation_dir}/{mechanism}/{args.imputer}/{missing_rate}/{filename}_data_imputed.npy', data_imputed)