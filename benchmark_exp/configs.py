from MissingModules.imputers.LERP import LERPimputer
from MissingModules.imputers.FillBy0 import FillBy0imputer
from MissingModules.imputers.SAITS import SAITSConfig
from benchmark_exp.configs_local import DEFAULT_PATHS

SEED = 2024
MISSING_RATES = [0., 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
MECHANISMS = ['mcar']

IMPUTERS =  {
            'LERP': LERPimputer(), 
            #'FillBy0': FillBy0imputer(),
            #'SAITS': SAITSConfig()
            }