from TSB_AD.missing_modules.imputers.LERP import LERPimputer
from TSB_AD.missing_modules.imputers.FillBy0 import FillBy0imputer
from TSB_AD.missing_modules.imputers.SAITS import SAITSConfig
from benchmark_exp.configs_local import DELL_PATHS as DEFAULT_PATHS

SEED = 2024
MISSING_RATES = [0., 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
MECHANISMS = ['mcar']

IMPUTERS =  {
            'LERP': LERPimputer(), 
            #'FillBy0': FillBy0imputer(),
            #'SAITS': SAITSConfig()
            }