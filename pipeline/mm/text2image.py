from utils import FrameWorkType
from ..base import BasePipelineWrapper, BasePtPipeline


class TextImagePipelineWrapper(BasePipelineWrapper):
    def __init__(self, default_framework: FrameWorkType = None,
                 default_mode: str = None,
                 **kwargs):
        super().__init__(default_framework, default_mode, **kwargs)
    
    def __call__(self, prompt, **kwargs):
        width = kwargs.get("width", 1280)
        height = kwargs.get("height", 720)

        params = "input: {}, width: {}, height: {}".format(prompt, width, height)

        print(params)

        prompt = self._preprocess(prompt)
        print(prompt)

        resp = self._forward(prompt, width=width, height=height)
        print("forward", resp)

        resp = self._postprocess(resp)
        print("postprocess", resp)
        return resp
    
    def _preprocess(self, prompt, **kwargs):
        return self.pipeline.preprocess(prompt, **kwargs)
    
    def _forward(self, prompt, **kwargs):
        return self.pipeline.forward(prompt, **kwargs)
    
    def _postprocess(self, resp, **kwargs):
        return self.pipeline.postprocess(resp, **kwargs)
    

class PtText2ImagePipeline(BasePtPipeline):
    def __init__(self, task_name=None, model=None, requirements=None, **kwargs):
        super().__init__(task_name, model, requirements, **kwargs)
        self.template = None

    def init_and_load(self, model, config, **kwargs):
        super().init_and_load(model, config, **kwargs)
        self.model_id_or_path = model
        backend = kwargs.get("backend", None)
        print("PtText2ImagePipeline init_and_load {}".format(self.model_id_or_path))
        print("import diffuers")
        if backend == "silicon":
            print("import onediffx")

    def preprocess(self, prompt, **kwargs):
        return "preprocess" + prompt
    
    def forward(self, prompt, **kwargs):
        return ["pic1"]
    
    def postprocess(self, resp, **kwargs):
        return resp

