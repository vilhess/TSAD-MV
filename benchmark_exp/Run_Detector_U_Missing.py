import pandas as pd
import numpy as np
import torch
import random, argparse, time, os, logging
import gc

from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.model_wrapper import *
from TSB_AD.HP_list import Optimal_Uni_algo_HP_dict

from TSB_AD.missing_modules.masker import get_mask

from benchmark_exp.configs import MISSING_RATES, MECHANISMS, IMPUTERS, DEFAULT_PATHS
from benchmark_exp.utils import set_seed, print_cuda_info
from benchmark_exp.data_loader import load_file, train_split, get_dict_data

# seeding
seed = 2024
set_seed(seed)

print_cuda_info()

if __name__ == '__main__':

    Start_T = time.time()
    ## ArgumentParser
    parser = argparse.ArgumentParser(description='Generating Anomaly Score')
    parser.add_argument('--dataset_dir', type=str, default=DEFAULT_PATHS['dataset_dir'])
    parser.add_argument('--file_list', type=str, default=DEFAULT_PATHS['file_list'])
    parser.add_argument('--save_dir', type=str, default=DEFAULT_PATHS['save_dir'])
    parser.add_argument('--AD_Name', type=str, default='TranAD')
    parser.add_argument('--load_imputation_dir', type=str, default=DEFAULT_PATHS['load_imputation_dir'])
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok = True)

    file_list = pd.read_csv(args.file_list)['file_name'].values
    Optimal_Det_HP = Optimal_Uni_algo_HP_dict[args.AD_Name]

    all_writes_csv = {mechanism: {imputer.imputer_name: {missing_rate: [] for missing_rate in MISSING_RATES} for imputer in IMPUTERS.values()} for mechanism in MECHANISMS}

    col_w = None  # set on first successful get_metrics call (or from loaded CSV when resuming)

    #start_at = "810_Exathlon_id_1_Facility_tr_10766_1st_12590.csv"
    #if start_at in file_list:
    #    file_list = file_list[file_list.tolist().index(start_at):]
    #    for mechanism in MECHANISMS:
    #        for imputer in IMPUTERS.values():
    #            for missing_rate in MISSING_RATES:
    #                curr_csv = pd.read_csv(
    #                    f'{args.save_dir}/{mechanism}/{args.AD_Name}/{imputer.imputer_name}/{missing_rate}.csv'
    #                )
    #                curr_csv = curr_csv[~curr_csv['file'].isin(file_list)]
    #                all_writes_csv[mechanism][imputer.imputer_name][missing_rate] = curr_csv.values.tolist()
    #                if col_w is None and not curr_csv.empty:
    #                    col_w = list(curr_csv.columns)
    #    logging.info(f"Starting at {start_at}")
    #else:
    #    assert False, f"{start_at} not found in file list."

    # Pre-create output directories (fixed per run, no need to repeat inside the file loop)

    for mechanism in MECHANISMS:
        for imputer in IMPUTERS.values():
            os.makedirs(f'{args.save_dir}/{mechanism}/{args.AD_Name}/{imputer.imputer_name}', exist_ok=True)

    # Load existing results so we don't overwrite them on resume
    for mechanism in MECHANISMS:
        for imputer in IMPUTERS.values():
            for missing_rate in MISSING_RATES:
                save_path = f'{args.save_dir}/{mechanism}/{args.AD_Name}/{imputer.imputer_name}/{missing_rate}.csv'
                if os.path.exists(save_path):
                    existing = pd.read_csv(save_path)
                    all_writes_csv[mechanism][imputer.imputer_name][missing_rate] = existing.values.tolist()
                    if col_w is None and not existing.empty:
                        col_w = list(existing.columns)

    for i, filename in enumerate(file_list):

        print('Processing:{} by {}'.format(filename, args.AD_Name))

        file_path = os.path.join(args.dataset_dir, filename)
        data, label, slidingWindow = load_file(file_path)
        data_train = train_split(data, filename)

        clf = None

        for imputer in IMPUTERS.values():
            if getattr(imputer, 'trainable', False):
                if args.load_imputation_dir is None:
                    imputer.fit(data_train, prc_val=0.2)

        for mechanism in MECHANISMS:
            for imputer in IMPUTERS.values():
                for missing_rate in MISSING_RATES:

                    if any(row[0] == filename for row in all_writes_csv[mechanism][imputer.imputer_name][missing_rate]):
                        print(f"Already processed {filename} for {mechanism} with {imputer.imputer_name} at missing rate {missing_rate}, skipping.")
                        continue

                    mask = get_mask(torch.from_numpy(data), seed=seed*i*(1+10*missing_rate), missing_rate=missing_rate, strategy=mechanism).numpy()
                    all_datas = get_dict_data(data_train, data, mask, imputer, missing_rate, args.load_imputation_dir, filename, mechanism)

                    start_time = time.time()

                    if args.AD_Name in Semisupervise_AD_Pool:
                        output, clf = run_Semisupervise_AD(args.AD_Name, data_train, all_datas, clf, **Optimal_Det_HP)
                    elif args.AD_Name in Unsupervise_AD_Pool:
                        output, _ = run_Unsupervise_AD(args.AD_Name, all_datas, clf, **Optimal_Det_HP)
                    else:
                        raise Exception(f"{args.AD_Name} is not defined")

                    run_time = time.time() - start_time

                    try:
                        evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow)
                        print('evaluation_result: ', evaluation_result)
                        metrics = list(evaluation_result.values())
                        if col_w is None:
                            col_w = ['file', 'Time'] + list(evaluation_result.keys())
                    except Exception:
                        metrics = [0] * 9

                    row = [filename, run_time] + metrics
                    all_writes_csv[mechanism][imputer.imputer_name][missing_rate].append(row)

                    ## Temp Save
                    if col_w is not None:
                        save_path = f'{args.save_dir}/{mechanism}/{args.AD_Name}/{imputer.imputer_name}/{missing_rate}.csv'
                        pd.DataFrame(
                            all_writes_csv[mechanism][imputer.imputer_name][missing_rate],
                            columns=col_w,
                        ).to_csv(save_path, index=False)
        
    del output, all_datas, data, label, mask, clf, imputer
    torch.cuda.empty_cache()
    gc.collect()


        
    