from abc import ABC
from typing import List

from utils.framework_type import FrameWorkType


class BasePipelineWrapper(ABC):
    def __init__(self,
                 default_framework: str = None,
                 default_model: str = None,
                 **kwargs):
        self.default_framework = default_framework
        self.default_model = default_model
        self.pipeline = None
        self.framework = None

    def set_pipeline(self, pipeline):
        self.pipeline = pipeline
        self.framework = pipeline.framework
    
    def __call__(self, **kwargs):
        raise NotImplementedError
    
    def _preprocess(self, **kargs):
        raise NotImplementedError
    
    def _forward(self, **kwargs):
        raise NotImplementedError
    
    def _postprocess(self, **kwargs):
        raise NotImplementedError
    

class BasePipeline(ABC):
    def __init__(self,
                 task_name: str = None,
                 model_id_or_path: str = None,
                 requirements: List[str] = None,
                 **kwargs):
        self.task_name = task_name
        self.model_id_or_path = model_id_or_path
        self.requirements = requirements
        self.framework = None

    def init_and_load(self, model, config, **kwargs):
        raise NotImplementedError
    
    def preprocess(self, **kwargs):
        raise NotImplementedError
    
    def postprocess(self, **kwargs):
        raise NotImplementedError
    

class BasePtPipeline(BasePipeline):
    def __init__(self,
                 task_name: str = None,
                 model_id_or_path: str = None,
                 requirements: List[str] = None,
                 **kwargs):
        super().__init__(task_name, model_id_or_path, requirements, **kwargs)
        self.framework = FrameWorkType.PT

    def init_and_load(self, model, config, **kwargs):
        self.model_id_or_path = model

class BaseMsPipeline(BasePipeline):
    def __init__(self,
                 task_name: str = None,
                 model_id_or_path: str = None,
                 requirements: List[str] = None,
                 **kwargs):
        super().__init__(task_name, model_id_or_path, requirements, **kwargs)
        self.framework = FrameWorkType.MS

    def init_and_load(self, model, config, **kwargs):
        self.model_id_or_path = model
