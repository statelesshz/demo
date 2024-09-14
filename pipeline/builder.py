from typing import Dict, Optional, Literal

from task import TaskName
from utils.framework_type import FrameWorkType
from .base import BasePipeline
from .common import *
from .mm.text2image import TextImagePipelineWrapper, PtText2ImagePipeline
from .nlp.chat import LlmChatPipelineWrapper, MsLlmChatPipeline


PIPELINE_WRAPPER_MAPPING: Dict[str, BasePipelineWrapper] = {}
PIPELINE_MAPPING: Dict[str, Dict[FrameWorkType, BasePipeline]] = {}


class DefaultPipelineMode:
    PT_CHAT_MODEL = "PyTorch/chatglm-6b"


class Requirements:
    TRANSFORMERS = "transformers"
    PEFT = "peft"
    MINDFORMERS = "mindformers"
    DIFFUERS = "diffusers"
    MINDONE = "mindone"


def register_pipeline_wrapper(task_name: str, pipeline_wrapper: BasePipelineWrapper, **kwargs) -> None:
    pipeline_wrapper.task = task_name
    PIPELINE_WRAPPER_MAPPING[task_name] = pipeline_wrapper


def register_pipeline(task_name: str, framework: FrameWorkType, pipeline: BasePipeline, **kwargs) -> None:
    if task_name not in PIPELINE_MAPPING:
        PIPELINE_MAPPING[task_name] = {}
    pipeline.task_name = task_name
    PIPELINE_MAPPING[task_name][framework] = pipeline


def dummy_model_2_task(model_id_or_path: str) -> str:
    mapping = {
        "Baichuan/Baichuan2-7b-chat": "chat",
        "sd/stable-diffusion-v1.5": "text2image",
    }
    if model_id_or_path in mapping:
        return mapping[model_id_or_path]
    else:
        return TaskName.common
    
def get_pipeline_wrapper(
        task=None,
        model=None,
        config=None,
        tokenizer=None,
        feature_extractor=None,
        image_processor=None,
        framework: Optional[Literal["pt", "ms"]] = None,
        **kwargs
):
    if task is None and model is not None:
        task = dummy_model_2_task(model)
    pipeline_wrapper = PIPELINE_WRAPPER_MAPPING[task]
    print(pipeline_wrapper.default_framework)

    if framework is None and pipeline_wrapper.default_framework:
        framework = pipeline_wrapper.default_framework
    elif type(framework) is str:
        framework = FrameWorkType[framework.upper()]
    else:
        framework = FrameWorkType.PT

    if model is None:
        model = pipeline_wrapper.default_model
    
    if model is None:
        raise KeyError(f"no default model supported for {task} task")
    if framework not in PIPELINE_MAPPING[task]:
        raise KeyError(f"{framework} not supported for {task} task")
    
    pipeline = PIPELINE_MAPPING[task][framework]
    
    pipeline.init_and_load(model, config, **kwargs)
    pipeline_wrapper.set_pipeline(pipeline)
    return pipeline_wrapper


register_pipeline_wrapper(TaskName.llm_chat,
                          LlmChatPipelineWrapper(FrameWorkType.PT, DefaultPipelineMode.PT_CHAT_MODEL))
register_pipeline_wrapper(TaskName.text2image,
                          TextImagePipelineWrapper)
register_pipeline_wrapper(TaskName.common, CommonPipelineWrapper())

register_pipeline(TaskName.llm_chat, FrameWorkType.MS, MsLlmChatPipeline(requirements=[Requirements.MINDFORMERS]))
register_pipeline(TaskName.llm_chat, FrameWorkType.PT, MsLlmChatPipeline(requirements=[Requirements.TRANSFORMERS]))
register_pipeline(TaskName.text2image, FrameWorkType.PT, PtText2ImagePipeline(requirements=[Requirements.DIFFUERS]))
