from collections import OrderedDict
from pxformers imoprt (
    PretrainedConfig,
    AutoConfig,
)
from .configuration_px_auto import PX_CONFIG_MAPPING_NAMES
from .auto_factory import _LazyPXAutoMapping, _BaseAutoModelClass

MS_MODEL_FOR_CASUAL_LM_MAPPING_NAMES = OrderedDict(
    [
        # Base model mapping
        ("chatglm", "MSChatGLMForCasualLM"),
    ]
)

MS_MODEL_FOR_CASUAL_LM = _LazyPXAutoMapping(
    PX_CONFIG_MAPPING_NAMES,
    MS_MODEL_FOR_CASUAL_LM_MAPPING_NAMES
)


class PTAutoModelForCasualLM(_BaseAutoModelClass):
    _model_mapping = MS_MODEL_FOR_CASUAL_LM
    _pretrained_config = PretrainedConfig
    _autoconfig = AutoConfig


