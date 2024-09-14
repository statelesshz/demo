from ..base import BasePipelineWrapper, BaseMsPipeline, BasePtPipeline
from models import OP_MODEL_MAPPING, MODEL_ID2_OP_MAPPING
from utils import FrameWorkType


class LlmChatPipelineWrapper(BasePipelineWrapper):
    def __init__(self, default_framework: FrameWorkType = None,
                 default_model: str = None,
                 **kwargs):
        super().__init__(default_framework, default_model, **kwargs)
    
    def __call__(self, prompt, top_k=10, **kwargs):
        # 模拟不同框架的参数转换和适配
        prompt = "wrapper prompt of {}".format(self.framework.name)

        print(prompt)

        prompt = self._preprocess(prompt)
        print(prompt)

        resp = self._forward(prompt, **kwargs)
        print(resp)

        resp = self._postprocess(resp)
        return resp

    def _preprocess(self, prompt, **kwargs):
        return self.pipeline.preprocess(prompt, **kwargs)
    
    def _forward(self, prompt, **kwargs):
        return self.pipeline.forward(prompt, **kwargs)
    
    def _propcess(self, resp, **kwargs):
        return self.pipeline.postprocess(resp, **kwargs)
    

class MsLlmChatPipeline(BaseMsPipeline):
    def __init__(self, task_name=None, model=None, requirements=None, **kwargs):
        super().__init__(task_name, model, requirements, **kwargs)
        self.template = None

    def init_and_load(self, model, config, **kwargs):
        super().init_and_load(model, config, **kwargs)

        if self.model_id_or_path in MODEL_ID2_OP_MAPPING:
            op_model_name = MODEL_ID2_OP_MAPPING[self.model_id_or_path]
            self.template = OP_MODEL_MAPPING[op_model_name].template

        print("MsLlmChatPipeline init-and-load {}".format(self.model_id_or_path))
        print("import mindformers")

    def preprocess(self, prompt, **kwargs):
        if self.template is not None:
            return "preprocess" + self.template.encode(prompt)
        else:
            return "preprocess" + prompt
        
    def forward(self, prompt, **kwargs):
        return "forward"
    
    def postprocess(self, resp, **kwargs):
        return "postprocess" + resp
    

class PtLlmChatPipeline(BaseMsPipeline):
    def __init__(self, task_name=None, model=None, requirements=None, **kwargs):
        super().__init__(task_name, model, requirements, **kwargs)
        self.template = None

    def init_and_load(self, model, config, **kwargs):
        super().init_and_load(model, config, **kwargs)

        if self.model_id_or_path in MODEL_ID2_OP_MAPPING:
            op_model_name = MODEL_ID2_OP_MAPPING[self.model_id_or_path]
            self.template = OP_MODEL_MAPPING[op_model_name].template

        print("PtLlmChatPipeline init-and-load {}".format(self.model_id_or_path))
        print("import transformers")

    def preprocess(self, prompt, **kwargs):
        if self.template is not None:
            return "preprocess" + self.template.encode(prompt)
        else:
            return "preprocess" + prompt
        
    def forward(self, prompt, **kwargs):
        return "forward"
    
    def postprocess(self, resp, **kwargs):
        return "postprocess" + resp
