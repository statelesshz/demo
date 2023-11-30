from px.utils import backend

if backend == "pt":
    from .configuration_hf_auto import HFAutoConfig
    AutoConfig = HFAutoConfig
else:
    from .configuration_px_auto import PXAutoConfig
    AutoConfig = PXAutoConfig 