from datasets import load_dataset
from huggingface_hub import login

dataset = load_dataset("imagefolder", data_dir="ff25", drop_labels=False)

access_token = "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"

login(token=access_token)

dataset.push_to_hub("KevinNotSmile/private_ff25", private=True, token=access_token)