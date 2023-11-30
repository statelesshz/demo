from transformers import CONFIG_MAPPING_NAMES, PretrainedConfig
from .configuration_px_auto import PX_CONFIG_MAPPING

from .auto_factory import _LazyMixinConfigMapping, _BaseAutoConfig

HF_CONFIG_MAPPING = _LazyMixinConfigMapping(CONFIG_MAPPING_NAMES, PX_CONFIG_MAPPING)

class HFAutoConfig(_BaseAutoConfig):
    _config_mapping = HF_CONFIG_MAPPING
    _pretrained_config = PretrainedConfig