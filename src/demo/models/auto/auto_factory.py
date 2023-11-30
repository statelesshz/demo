import importlib
from collections import OrderedDict
import warnings

from px.dynamic_module_utils import get_class_from_dynamic_module, resolve_trust_remote_code
from px.utils imoprt CONFIG_NAME

# 
# config
#

def model_type_to_module_name(key):
    """Converts a config key to the corresponding module."""
    return key.replace("-", "_")

class _LazyPXConfigMapping(OrderedDict):
    """
    A dictionary that lazily load its values when they are requested.
    """

    def __init__(self, mapping):
        self._mapping = mapping
        self._extra_content = {}
        self._modules = {}

    def __getitem__(self, key):
        if key in self._extra_content:
            return self._extra_content[key]
        if key not in self._mapping:
            raise KeyError(key)
        value = self._mapping[key]
        module_name = model_type_to_module_name(key)
        if module_name not in self._modules:
            self._modules[module_name] = importlib.import_module(f".{module_name}", "px.models")
        if hasattr(self._modules[module_name], value):
            return getattr(self._modules[module_name], value)

        # Some of the mappings have entries model_type -> config of another model type. In that case we try to grab the
        # object at the top level.
        px_module = importlib.import_module("px")
        return getattr(px_module, value)

    def keys(self):
        return list(self._mapping.keys()) + list(self._extra_content.keys())

    def values(self):
        return [self[k] for k in self._mapping.keys()] + list(self._extra_content.values())

    def items(self):
        return [(k, self[k]) for k in self._mapping.keys()] + list(self._extra_content.items())

    def __iter__(self):
        return iter(list(self._mapping.keys()) + list(self._extra_content.keys()))

    def __contains__(self, item):
        return item in self._mapping or item in self._extra_content

    def register(self, key, value, exist_ok=False):
        """
        Register a new configuration in this mapping.
        """
        if key in self._mapping.keys() and not exist_ok:
            raise ValueError(f"'{key}' is already used by a Transformers config, pick another name.")
        self._extra_content[key] = value


class _LazyMixinConfigMapping(OrderedDict):
    """
    A dictionary that lazily load its values when they are requested.
    """

    def __init__(self, px_mapping, hf_mapping):
        self._px_mapping = px_mapping
        self._hf_mapping = hf_mapping
        self._extra_content = {}
        self._modules = {}

    def __getitem__(self, key):
        pass

    def keys(self):
        pass

    def values(self):
        pass

    def items(self):
        pass

    def __iter__(self):
        pass

    def __contains__(self, item):
        pass

    def register(self, key, value, exist_ok=False):
        """
        Register a new configuration in this mapping.
        """
        pass


class _BaseAutoConfig:
    r"""
    base auto config class
    """
    _config_mapping = None
    _pretrained_config = None

    def __init__(self):
        raise EnvironmentError(
            "AutoConfig is designed to be instantiated "
            "using the `_BaseAutoConfig.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def for_model(cls, model_type: str, *args, **kwargs):
        if model_type in _config_mapping:
            config_class = _config_mapping[model_type]
            return config_class(*args, **kwargs)
        raise ValueError(
            f"Unrecognized model identifier: {model_type}. Should contain one of {', '.join(_config_mapping.keys())}"
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
       
        use_auth_token = kwargs.pop("use_auth_token", None)
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if kwargs.get("token", None) is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            kwargs["token"] = use_auth_token

        kwargs["_from_auto"] = True
        kwargs["name_or_path"] = pretrained_model_name_or_path
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        code_revision = kwargs.pop("code_revision", None)

        config_dict, unused_kwargs = _pretrained_config.get_config_dict(pretrained_model_name_or_path, **kwargs)
        has_remote_code = "auto_map" in config_dict and "AutoConfig" in config_dict["auto_map"]
        has_local_code = "model_type" in config_dict and config_dict["model_type"] in _config_mapping
        trust_remote_code = resolve_trust_remote_code(
            trust_remote_code, pretrained_model_name_or_path, has_local_code, has_remote_code
        )

        if has_remote_code and trust_remote_code:
            class_ref = config_dict["auto_map"]["AutoConfig"]
            config_class = get_class_from_dynamic_module(
                class_ref, pretrained_model_name_or_path, code_revision=code_revision, **kwargs
            )
            if os.path.isdir(pretrained_model_name_or_path):
                config_class.register_for_auto_class()
            return config_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
        elif "model_type" in config_dict:
            config_class = _config_mapping[config_dict["model_type"]]
            return config_class.from_dict(config_dict, **unused_kwargs)
        else:
            # Fallback: use pattern matching on the string.
            # We go from longer names to shorter names to catch roberta before bert (for instance)
            for pattern in sorted(_config_mapping.keys(), key=len, reverse=True):
                if pattern in str(pretrained_model_name_or_path):
                    return _config_mapping[pattern].from_dict(config_dict, **unused_kwargs)

        raise ValueError(
            f"Unrecognized model in {pretrained_model_name_or_path}. "
            f"Should have a `model_type` key in its {CONFIG_NAME}, or contain one of the following strings "
            f"in its name: {', '.join(_config_mapping.keys())}"
        )

    @staticmethod
    def register(model_type, config, exist_ok=False):
        """
        Register a new configuration for this class.
        """
        if issubclass(config, _pretrained_config) and config.model_type != model_type:
            raise ValueError(
                "The config you are passing has a `model_type` attribute that is not consistent with the model type "
                f"you passed (config has {config.model_type} and you passed {model_type}. Fix one of those so they "
                "match!"
            )
        _config_mapping.register(model_type, config, exist_ok=exist_ok)



#
# model
#
class _BaseAutoModelClass:
    # Base class for auto models.
    # 
    _model_mapping = None  # _LazyAutoMapping(config_mapping_name, model_mapping_name)
    _pretrained_config = None
    _autoconfig = None

    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)` or "
            f"`{self.__class__.__name__}.from_config(config)` methods."
        )

    @classmethod
    def from_config(cls, config, **kwargs):
        pass

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
       pass

    @classmethod
    def register(cls, config_class, model_class, exist_ok=False):
        pass


class _LazyPXAutoMapping(OrderedDict):  # 对外呈现的结果是 XXXConfig->XXXModelxxx
    def __init__(self, config_mapping, model_mapping):
        self._config_mapping = config_mapping  # model_type -> XXXConfig
        self._reverse_config_mapping = {v: k for k, v in config_mapping.items()}# XXXConfig -> model_type
        self._model_mapping = model_mapping   # model_type -> XXXModel
        self._extra_content = {} # 外部注册的 XXXConfig->XXXModel
        self._modules = {} # module_name->module =>  'albert': <module 'transformers.models.albert' from '/Users/yun/github/transformers/src/transformers/models/albert/__init__.py'>

    def __len__(self):
        pass
    def __getitem__(self, key):
        pass

    def _load_attr_from_module(self, model_type, attr):
       pass

    def keys(self):
        pass

    def get(self, key, default):
        pass

    def __bool__(self):
        pass

    def values(self):
        pass

    def items(self):
        pass

    def __iter__(self):
        return iter(self.keys())

    def __contains__(self, item):
        pass

    def register(self, key, value, exist_ok=False):
        pass


class _LazyMixinAutoMapping(OrderedDict):  # 对外呈现的结果是 XXXConfig->XXXModelxxx
    def __init__(self, hf_config_mapping, hf_model_mapping, px_config_mapping, px_model_mapping):
        pass
    
    def __len__(self):
        pass
    def __getitem__(self, key):
        pass

    def _load_attr_from_module(self, model_type, attr):
       pass

    def keys(self):
        pass

    def get(self, key, default):
        pass

    def __bool__(self):
        pass

    def values(self):
        pass

    def items(self):
        pass

    def __iter__(self):
        return iter(self.keys())

    def __contains__(self, item):
        pass

    def register(self, key, value, exist_ok=False):
        pass