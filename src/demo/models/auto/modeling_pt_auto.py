from collections import OrderedDict
from transformers imoprt (
    CONFIG_MAPPING_NAME,
    MODEL_FOR_CASUAL_LM__MAPPING_NAMES,
    PretrainedConfig,
    AutoConfig,
)
from .configuration_px_auto import PX_CONFIG_MAPPING_NAMES
from .auto_factory import _LazyMixinAutoMapping, _BaseAutoModelClass

PT_MODEL_FOR_CASUAL_LM_MAPPING_NAMES = OrderedDict(
    [
        # Base model mapping
        ("chatglm", "ChatGLMForCasualLM"),
    ]
)

PT_MODEL_FOR_CASUAL_LM = _LazyMixinAutoMapping(
    CONFIG_MAPPING_NAME,
    MODEL_FOR_CASUAL_LM__MAPPING_NAMES,
    PX_CONFIG_MAPPING_NAMES,
    PT_MODEL_FOR_CASUAL_LM_MAPPING_NAMES)


class PTAutoModelForCasualLM(_BaseAutoModelClass):
    _model_mapping = PT_MODEL_FOR_CASUAL_LM
    _pretrained_config = PretrainedConfig
    _autoconfig = AutoConfig


