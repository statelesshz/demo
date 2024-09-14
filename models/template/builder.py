from typing import Dict

from .base import BaseTemplate

TEMPLATE_MAPPING: Dict[str, BaseTemplate] = {}


class TemplateName:
    LLAMA = "llama"
    QWEN2_VL = "qwen2vl"


class LLamaTemplate(BaseTemplate):
    def __init__(self):
        prefix = "llama template"
        super().__init__(prefix)

    def encode(self, prompt):
        return self.prefix + prompt
    

class BaseMMTemplate(BaseTemplate):
    def __init__(self, prefix, mm_plugin):
        super().__init__(prefix)
        self.mm_plugin = mm_plugin

    def encode(self):
        raise NotImplementedError
    

def register_templates(template_name, template: BaseTemplate):
    TEMPLATE_MAPPING[template_name] = template


register_templates(TemplateName.LLAMA, LLamaTemplate())
