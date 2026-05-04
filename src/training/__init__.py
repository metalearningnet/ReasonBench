from .lm import LMTrainer
from .utils import create_rl_trainer
from .dpo import RLDPOConfig, RLDPOTrainer, RLDPOPreprocessor
from .cpo import RLCPOConfig, RLCPOTrainer, RLCPOPreprocessor
from .kto import RLKTOConfig, RLKTOTrainer, RLKTOPreprocessor
from .orpo import RLORPOConfig, RLORPOTrainer, RLORPOPreprocessor

CONFIG_MAP = {
    'dpo': RLDPOConfig,
    'cpo': RLCPOConfig,
    'kto': RLKTOConfig,
    'orpo': RLORPOConfig
}

TRAINER_MAP = {
    'dpo': RLDPOTrainer,
    'cpo': RLCPOTrainer,
    'kto': RLKTOTrainer,
    'orpo': RLORPOTrainer
}

PREPROCESSOR_MAP = {
    'dpo': RLDPOPreprocessor,
    'cpo': RLCPOPreprocessor,
    'kto': RLKTOPreprocessor,
    'orpo': RLORPOPreprocessor
}
