from datasets import load_dataset
dataset = load_dataset("parquet", data_files={'train': './data/mrpc/train-00000-of-00001.parquet', 'test': './data/mrpc/validation-00000-of-00001.parquet'})

print(dataset)
