from typing import Dict

from .base import Task
from .domain_type import DomainType
from .input_output_type import InputAndOutputType

TASK_MAPPING: Dict[str, Task] = {}

class TaskName:
    text_classification = "text-classification"
    llm_chat = "chat"
    text2image = "text2image"
    mllm_chat = "mllm_chat"
    common = "common"


def register_task(task_name: str, task: Task, **kwargs) -> None:
    TASK_MAPPING[task_name] = task

register_task(TaskName.text_classification, Task(DomainType.NLP, InputAndOutputType.TEXT, InputAndOutputType.TEXT))
register_task(TaskName.llm_chat, Task(DomainType.NLP, InputAndOutputType.TEXT, InputAndOutputType.TEXT))
register_task(TaskName.text2image, Task(DomainType.MM, InputAndOutputType.TEXT, InputAndOutputType.IMAGE))
register_task(TaskName.mllm_chat, Task(DomainType.MM,
                                       [InputAndOutputType.TEXT, InputAndOutputType.IMAGE, InputAndOutputType.VIDEO, InputAndOutputType.AUDIO], InputAndOutputType.TEXT))
