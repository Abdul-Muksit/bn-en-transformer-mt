from datasets import load_dataset

ds = load_dataset("csebuetnlp/BanglaNMT", trust_remote_code=True)


print(ds['train'][0])