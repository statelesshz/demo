from px.utils import backend

if backend == "pt":
    from .modeling_pt_auto import PTAutoModelForCasualLM
    AutoModelForCasual = PTAutoModelForCasualLM
else:
    from .modeling_ms_auto import MSAutoModelForCasual
    AutoConfig = MSAutoModelForCasual 