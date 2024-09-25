import os
from pathlib import Path
from typing import List


def list_file_paths(model_dir):
  # local version
  return [str(p) for p in Path(model_dir).rglob("*")]


def check_framework(paths: List[str]):
  for path in paths:
    # 先判断是不是pytorch格式的，主要考虑hf transformers diffusers结构
    if path.endswith(".bin") or path.endswith(".safetensors"):
      return "torch"

  # fallback到mindspore
  return "mindspore"


def check_backend_and_model_type(paths: List[str], framework: str):
  # 获取backend & model_type
  if framework == "torch":
    # 先看是不是diffusers
    
  elif framework == "mindspore":
    ...
  for path in paths:
    if "diffusers" in path:
      return "diffusers"


def parse_model_metadata(model_name_or_path):
  # 尽力而为的解析
  is_local = os.path.isdir(model_name_or_path)

  if is_local:
    print("start scanning local file.")
  else:
    print("get model metadata from hub")
    # model_repo


if __name__ == "__main__":
  # print(parse_model_metadata("/home/lynn/github/qwen2.5-0.5b-instruct"))
  model_dir = "/home/lynn/github/qwen2.5-0.5b-instruct"
  print(list_file_paths(model_dir))
