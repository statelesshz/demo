from utils import FrameWorkType
from ..base import BasePipelineWrapper, BaseMsPipeline, BasePtPipeline

class CommonPipelineWrapper(BasePipelineWrapper):
    def __init__(self, default_framework: FrameWorkType = None,
                 default_model: str = None,
                 **kwargs):
        super().__init__(default_framework, default_model, **kwargs)
    
    def __call__(self, **kwargs):
        # 模拟不同框架参数转换和适配

        print(kwargs)

        # 预处理
        kwargs = self._preprocess(**kwargs)
        print(kwargs)

        # inference
        response = self._forward(**kwargs)
        print(response)

        #postprocess
        response = self._postprocess(response)
        return response
    
    def _preprocess(self, **kwargs):
        self.pipeline.preprocess(**kwargs)

    def _forward(self, **kwargs):
        return self.pipeline.forward(**kwargs)
    
    def _postprocess(self, response, **kwargs):
        return self.pipeline.postprocess(response, **kwargs)
    

class CommonPtPipeline(BasePtPipeline):
    def __init__(self, task_name=None, model=None, requirements=None, **kwargs):
        super().__init__(task_name, model, requirements, **kwargs)
        self.template = None

    def init_and_load(self, model, config, **kwargs):
        super().init_and_load(model, config, **kwargs)
        print("CommonPtPipeline init_and_load {}".format(self.model_id_or_path))

    def preprocess(self, **kwargs):
        kwargs["preprocess"] = 1
        return kwargs
    
    def postprocess(self, respone, **kwargs):
        return "postprocess" + respone
    

class CommonMsPipeline(BaseMsPipeline):
    def __init__(self, task_name=None, model=None, requirements=None, **kwargs):
        super().__init__(task_name, model, requirements, **kwargs)
        self.template = None

    def init_and_load(self, model, config, **kwargs):
        super().init_and_load(model, config, **kwargs)
        print("CommonMsPipeline init_and_load {}".format(self.model_id_or_path))

    def preprocess(self, **kwargs):
        kwargs["preprocess"] = 1
        return kwargs
    
    def postprocess(self, respone, **kwargs):
        return "postprocess" + respone