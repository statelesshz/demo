import os


model_dir = "/home/lynn/github/"

is_local = os.path.isdir(model_dir)

dir_entries = [f for f in os.listdir(model_dir)]

if "config.json" in dir_entries:
  print("backend: transformers")

if "pytorch_model.bin" or "pytorch_model.bin.index.json" or "model.safetensors.index.json" or "model.safetensors" in dir_entries:
  print("famework: torch")
elif "mindspore_model.ckpt" or "mindspore_model.ckpt.index.json" in dir_entries:
  print("framework: mindspore")

# parse_model_metadata()


