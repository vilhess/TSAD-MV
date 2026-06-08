# -*- coding: utf-8 -*-
# Author: Qinghua Liu <liu.11085@osu.edu>
# License: Apache-2.0 License

import pandas as pd
import numpy as np
import torch
import random, argparse, time, os, logging

from sklearn.preprocessing import MinMaxScaler
from .evaluation.metrics import get_metrics
from .utils.slidingWindows import find_length_rank
from .model_wrapper import *
from .HP_list import Optimal_Uni_algo_HP_dict
from .HP_list import Optimal_Multi_algo_HP_dict
from .missing_modules.masker import get_mask
from .missing_modules.imputers.LERP import LERPConfig as LERP

# seeding
seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

print("CUDA Available: ", torch.cuda.is_available())
print("cuDNN Version: ", torch.backends.cudnn.version())

if __name__ == '__main__':

    ## ArgumentParser
    parser = argparse.ArgumentParser(description='Running TSB-AD')
    parser.add_argument('--filename', type=str, default='001_NAB_id_1_Facility_tr_1007_1st_2014.csv')
    parser.add_argument('--data_direc', type=str, default='Datasets/TSB-AD-U/')
    parser.add_argument('--save', type=bool, default=False)
    parser.add_argument('--AD_Name', type=str, default='patchtradmv')
    parser.add_argument('--missing_rate', type=float, default=0.1,
                        help='Missing rate to simulate (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5)')
    args = parser.parse_args()

    df = pd.read_csv(args.data_direc + args.filename).dropna()
    data = df.iloc[:, 0:-1].values.astype(float)
    label = df['Label'].astype(int).to_numpy()

    n_dim = data.shape[1]
    if n_dim == 1:
        slidingWindow = find_length_rank(data, rank=1)
    else:
        slidingWindow = find_length_rank(data[:, 0].reshape(-1, 1), rank=1)

    train_index = args.filename.split('.')[0].split('_')[-3]
    data_train = data[:int(train_index), :]

    if n_dim == 1:
        Optimal_Det_HP = Optimal_Uni_algo_HP_dict[args.AD_Name]
    else:
        Optimal_Det_HP = Optimal_Multi_algo_HP_dict[args.AD_Name]

    # --- Missing data simulation & imputation ---
    imputer = LERP()
    clf = None

    mask = get_mask(
        torch.from_numpy(data),
        seed=seed * (1 + 10 * args.missing_rate),
        missing_rate=args.missing_rate
    ).numpy()
    print(f"Target missing rate: {args.missing_rate}, "
          f"Observed missing rate: {mask.sum() / mask.size:.2%}")

    data_missing = data.copy()
    data_missing[mask == 1] = np.nan
    print(f"Data with missing values has "
          f"{np.isnan(data_missing).sum() / data_missing.size:.2%} missing entries.")

    data_imputed = imputer.impute(data_missing, mask)

    all_datas = {
        "data": data,
        "data_missing": data_missing,
        "data_imputed": data_imputed,
        "mask": mask
    }
    # --------------------------------------------

    start_time = time.time()

    if args.AD_Name in Semisupervise_AD_Pool:
        output, clf = run_Semisupervise_AD(args.AD_Name, data_train, all_datas, clf, **Optimal_Det_HP)
    elif args.AD_Name in Unsupervise_AD_Pool:
        output, _ = run_Unsupervise_AD(args.AD_Name, all_datas, clf, **Optimal_Det_HP)
    else:
        raise Exception(f"{args.AD_Name} is not defined")

    run_time = time.time() - start_time
    print(f"Run time: {run_time:.3f}s")

    if isinstance(output, np.ndarray):
        output = MinMaxScaler(feature_range=(0, 1)).fit_transform(output.reshape(-1, 1)).ravel()
        evaluation_result = get_metrics(
            output, label,
            slidingWindow=slidingWindow,
            pred=output > (np.mean(output) + 3 * np.std(output))
        )
        print('Evaluation Result: ', evaluation_result)

        if args.save:
            save_path = f'eval/scores/{args.AD_Name}/{type(imputer).__name__}/{args.missing_rate}/'
            os.makedirs(save_path, exist_ok=True)
            np.save(save_path + args.filename.split('.')[0] + '.npy', output)
            print(f"Score saved to {save_path}")
    else:
        print(f'At {args.filename}: ' + output)