from typing import Literal, Optional

from .builder import get_pipeline_wrapper


def pipeline(
        task: Optional[str] = None,
        model=None,
        config=None,
        tokenizer=None,
        feature_extractor=None,
        image_processor=None,
        framework: Optional[Literal["pt", "ms"]]=None,
        **kwargs,
):
    pipeline_wrapper = get_pipeline_wrapper(task, model, config, tokenizer, feature_extractor, image_processor, framework, **kwargs)
    return pipeline_wrapper
