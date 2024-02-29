import os
from common import execute_subprocess_async

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

MODEL_TO_PATH = dict()
MODEL_TO_TASK = dict()
TASK_TO_MODEL = dict()

MODELS_FILENAME = os.path.join(os.path.dirname(__file__), "huggingface_models_lists.txt")
assert os.path.exists(MODELS_FILENAME)
with open(MODELS_FILENAME, "r") as fh:
    lines = fh.readlines()
    lines = [line.rstrip() for line in lines]
    for line in lines:
        model_name, model_path, task_name = line.split(",")
        if model_name not in MODEL_TO_PATH:
            MODEL_TO_PATH[model_name] = [model_path]
        else:
            MODEL_TO_PATH[model_name].append(model_path)
        MODEL_TO_TASK[model_name] = task_name
        TASK_TO_MODEL[task_name] = model_name


def run_text_classification():
    script_path = os.path.sep.join([
        ROOT_PATH, "text-classification", "run_glue.py"
    ])
    model_path = "distilbert/distilbert-base-uncased"
    # TODO: output_dir搞成能自动化创建&删除的形式
    output_dir = os.path.sep.join([
        ROOT_PATH, "tmp", model_path.split("/")[-1]
    ])
    cmd = [
        "torchrun",
        "--nproc_per_node=8",
        "--master_port=29501",
        script_path,
        "--model_name_or_path=" + model_path,
        "--output_dir=" + output_dir,
        "--overwrite_output_dir",
        "--train_file=./data/mrpc/train-00000-of-00001.parquet",
        "--validation_file=./data/mrpc/validation-00000-of-00001.parquet",
        "--do_train",
        "--do_eval",
        "--per_device_train_batch_size=4",
        "--max_steps=2000",
        "--learning_rate=2e-5",
        "--report_to=none",
        "--max_seq_length=128",
    ]
    execute_subprocess_async(cmd)


def main():
    run_text_classification()


if __name__ == "__main__":
    main()
