import contextlib
import os

import datasets
import evaluate
import transformers
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table

from openmind.hf.hf_utils import (
    AutoConfig,
    AutoModelForCausalLM,
    # AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from openmind.omdatasets.om_datasets import OmDataset


def patch_load_dataset():
    def wrapper(path, *args, **kwargs):
        if path == "EleutherAI/arithmetic":
            path = "humphrey007/arithmetic"
        return OmDataset.load_dataset(path, *args, **kwargs)
    return wrapper


def patch_hf_evaluate_load():
    def wrapper(path, **kwargs):
        if path == "exact_match":
            path = "./exact_match"
        breakpoint()
        return evaluate.load(path, **kwargs)
    return wrapper


@contextlib.contextmanager
def do_eval_with_ctx(*args, **kwargs):
    # origin
    origin_env_tokenizers_parallelism = os.environ.get("TOKENIZERS_PARALLELISM")
    origin_hf_evaluate_load = evaluate.load
    origin_load_dataset = datasets.load_dataset
    origin_AutoConfig = transformers.AutoConfig
    origin_AutoTokenizer = transformers.AutoTokenizer
    origin_AutoModelForCausalLM = transformers.AutoModelForCausalLM
    # origin_AutoModelForSeq2SeqLM = transformers.AutoModelForSeq2SeqLM
    origin_get_model_info = HFLM.get_model_info
    origin_get_git_commit_hash = evaluator.get_git_commit_hash

    # patch
    os.environ["TOKENIZERS_PARALLELISM"] = "0"
    evaluate.load = patch_hf_evaluate_load()
    datasets.load_dataset = patch_load_dataset()
    transformers.AutoConfig = AutoConfig
    transformers.AutoTokenizer = AutoTokenizer
    transformers.AutoModelForCausalLM = AutoModelForCausalLM
    # transformers.AutoModelForSeq2SeqLM = AutoModelForSeq2SeqLM
    HFLM.get_model_info = lambda x: ""
    evaluator.get_git_commit_hash = lambda : ""


    try:
        result = evaluator.simple_evaluate(*args, **kwargs)
        print(make_table(result))
        if "groups" in result:
            print(make_table(result, "groups"))
        yield result
    finally:
        # rollback patch
        if origin_env_tokenizers_parallelism is not None:
            os.environ["TOKENIZERS_PARALLELISM"] = origin_env_tokenizers_parallelism
        else:
            del os.environ["TOKENIZERS_PARALLELISM"]
        evaluate.load = origin_hf_evaluate_load
        datasets.load_dataset =origin_load_dataset
        transformers.AutoConfig = origin_AutoConfig
        transformers.AutoTokenizer = origin_AutoTokenizer
        transformers.AutoModelForCausalLM = origin_AutoModelForCausalLM
        # transformers.AutoModelForSeq2SeqLM = origin_AutoModelForSeq2SeqLM
        HFLM.get_model_info = origin_get_model_info
        evaluator.get_git_commit_hash = origin_get_git_commit_hash


def eval(*args, **kwargs):
    with do_eval_with_ctx(
        model="hf",
        model_args="pretrained=qwen2-0.5b-instruct",
        tasks=["arithmetic"],
        batch_size=64,
        device="cpu",
        limit=1,
        **kwargs) as eval_result:
        return eval_result

results = eval()

# print(make_table(results))
# if "groups" in results:
#     print(make_table(results, "groups"))