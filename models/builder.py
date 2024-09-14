from typing import Dict, NamedTuple

from task import TaskName
from .base import BaseModel, LlmModel, SDModel
from .template import TemplateName, TEMPLATE_MAPPING

OP_MODEL_MAPPING: Dict[str, BaseModel] = {}
MODEL_ID2_OP_MAPPING: Dict[str, str] = {}


class OpModelName(NamedTuple):
    LLAMA1 = "llama1"



class LoRATM(NamedTuple):
    llama = ['q_proj', 'k_proj', 'v_proj']


def register_model(op_model_name: str, base_model: BaseModel) -> None:
    OP_MODEL_MAPPING[op_model_name] = base_model
    MODEL_ID2_OP_MAPPING[base_model.model_id_or_path] = op_model_name


register_model(OpModelName.LLAMA1,
               LlmModel(TaskName.llm_chat, "dummy/llama1", 2, LoRATM.llama, TEMPLATE_MAPPING[TemplateName.LLAMA]))

