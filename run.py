import os
import sys
from common import execute_subprocess_async


def run_text_classification():
    script_path = os.path.sep.join([
        os.path.dirname(os.path.abspath(__file__)), "text-classification", "run_glue.py"
    ])
    cmd = [
        "torchrun",
        "--nproc_per_node=8",
        "--master_port=29501",
        script_path,
        "--model_name_or_path=distilbert/distilbert-base-uncased",
        "--output_dir=./tmp_text_classification",
        "--overwrite_output_dir",
        "--train_file=./data/mrpc/train-00000-of-00001.parquet",
        "--validation_file=./data/mrpc/validation-00000-of-00001.parquet",
        "--do_train",
        "--do_eval",
        "--per_device_train_batch_size=4",
        "--per_device_eval_batch_size=2",
        "--learning_rate=2e-5",
        "--report_to=none",
        "--max_seq_length=128",
    ]
    execute_subprocess_async(cmd)


def main():
    run_text_classification()


if __name__ == "__main__":
    main()
