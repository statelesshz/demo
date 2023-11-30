from collections import OrderedDict

from px import PretrainedConfig
from .auto_factory import _LazyPXConfigMapping, _BaseAutoConfig

PX_CONFIG_MAPPING_NAMES = OrderedDict(
    [
        # Add configs here
        ("chatglm", "ChatGLMConfig"),
    ]
)

PX_CONFIG_MAPPING = _LazyPXConfigMapping(PX_CONFIG_MAPPING_NAMES)

class PXAutoConfig(_BaseAutoConfig):
    _config_mapping = PX_CONFIG_MAPPING
    _pretrained_config = PretrainedConfig