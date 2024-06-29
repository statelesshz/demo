import contextlib
import os

import transformers
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table

from openmind.hf.hf_utils import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)


@contextlib.contextmanager
def do_eval_with_ctx(*args, **kwargs):
    # origin
    origin_env_tokenizers_parallelism = os.environ.get("TOKENIZERS_PARALLELISM")
    origin_get_model_info = HFLM.get_model_info
    origin_get_git_commit_hash = evaluator.get_git_commit_hash
    origin_AutoConfig = transformers.AutoConfig
    origin_AutoTokenizer = transformers.AutoTokenizer
    origin_AutoModelForCausalLM = transformers.AutoModelForCausalLM
    origin_AutoModelForSeq2SeqLM = transformers.AutoModelForSeq2SeqLM

    # patch
    os.environ["TOKENIZERS_PARALLELISM"] = "0"
    HFLM.get_model_info = lambda x: ""
    evaluator.get_git_commit_hash = lambda : ""
    transformers.AutoConfig = AutoConfig
    transformers.AutoTokenizer = AutoTokenizer
    transformers.AutoModelForCausalLM = AutoModelForCausalLM
    transformers.AutoModelForSeq2SeqLM = AutoModelForSeq2SeqLM


    try:
        result = evaluator.simple_evaluate(*args, **kwargs)
        yield result
    finally:
        # rollback patch
        if origin_env_tokenizers_parallelism is not None:
            os.environ["TOKENIZERS_PARALLELISM"] = origin_env_tokenizers_parallelism
        else:
            del os.environ["TOKENIZERS_PARALLELISM"]
        HFLM.get_model_info = origin_get_model_info
        evaluator.get_git_commit_hash = origin_get_git_commit_hash
        transformers.AutoConfig = origin_AutoConfig
        transformers.AutoTokenizer = origin_AutoTokenizer
        transformers.AutoModelForCausalLM = origin_AutoModelForCausalLM
        transformers.AutoModelForSeq2SeqLM = origin_AutoModelForSeq2SeqLM


def eval(*args, **kwargs):
    with do_eval_with_ctx(
        model="hf",
        model_args="pretrained=qwen2-0.5b-instruct",
        tasks=["arithmetic"],
        batch_size=64,
        device="npu:0",
        limit=10,
        **kwargs) as eval_result:
        return eval_result

results = eval()

print(make_table(results))
if "groups" in results:
    print(make_table(results, "groups"))
