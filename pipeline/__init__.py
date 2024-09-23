from typing import Optional, Literal


def pipeline(
  task: Optional[str] = None,
  model=None, # Optional[Union[str, "Model"]] = None
  config=None,
  tokenizer=None,
  feature_extractor=None,
  image_processor=None,
  framework: Optional[Literal["pt", "ms"]] = None,
  **kwargs,
):
    if task is None and model is None:
       raise RuntimeError(
        "Impossible to instantiate a pipeline without either a task or a model "
        "being specified. "
        "Please provide a task class or a model"
      )
    if model is None and tokenizer is not None:
        raise RuntimeError(
            "Impossible to instantiate a pipeline with tokenizer specified but not the model as the provided tokenizer"
            " may not be compatible with the default model."
        )
    if model is None and feature_extractor is not None:
        raise RuntimeError(
            "Impossible to instantiate a pipeline with feature_extractor specified but not the model as the provided feature_extractor"
            " may not be compatible with the default model."
        )
    if model is None and image_processor is not None:
        raise RuntimeError(
            "Impossible to instantiate a pipeline with image_processor specified but not the model as the provided image_processor"
            " may not be compatible with the default model."
        )

    if task is None and model is not None:
        # 仅考虑model为字符串（本地路径or远端模型仓id）以便简单的阐述设计思路
        if isinstance(model, str): # 如何获取model呢？
            ...
            
        

    # pipeline_wrapper = get_pipeline_wrapper(task, model, config, tokenizer, feature_extractor, image_processor, framework, **kwargs)
    # return pipeline_wrapper