# FreezeAsGuard: Mitigating Illegal Adaptation of Diffusion Models via Selective Tensor Freezing

## Introduction
This is the official code repository for the paper ["FreezeAsGuard: Mitigating Illegal Adaptation of Diffusion Models via Selective Tensor Freezing"](https://arxiv.org/pdf/2405.17472). FreezeAsGuard is a new technique for mitigating illegal adaptations of diffusion models by selectively freezing model tensors that are adaptation-critical for illegal domains but still retain the representation power on innocent domains.

## Requirement
Install all the required packages.
```
pip install -r requirements.txt
```

## General Usage

### Download the FF25 Dataset
Download our `FamousFigures-25 (FF25)` dataset via the [anonymous link](https://pub-9dad475c5318471bb9941734361d67d5.r2.dev/ff25.zip). Unzip it and put the dataset folder `ff25` under the main directory. Now the dataset has been created locally and can be loaded by huggingface datasets API:
```
dataset = load_dataset("imagefolder", data_dir="ff25", split="train", drop_labels=False)
```

### Mask Learning

Optional: Navigate to `uce_tools` and open `run.sh`. Specify the target persons' name and a general concept for replacement. For example:
```
python3 train-scripts/train_erase.py \
    --concepts 'donald trump, emma watson' \
    --guided_concept 'celebrity' \
    --device '0' \
    --concept_type 'celebrity' \
    --save_path '../sd15_target_unlearned'
```
We consider the model after such erasing as the pretrained model we want. Of course you can opt to skip this step.

To run the mask learning algorithm, we need to first train a relearned model. Navigate to `src/` and run the following script: 
```
bash run_mixed_full_trainer.sh
```
Make sure to pass correct parameters (such as if including innocent representative) in the script. Here is an example of `run_mixed_full_trainer.sh`:
```
accelerate launch mixed_full_trainer.py \
  --pretrained_model_name_or_path="../sd15_target_unlearned" \
  --target_dataset_name="../ff25" \
  --innocent_dataset_name="logo-wizard/modern-logo-dataset" \
  --caption_column="text" \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --max_train_steps=2000 \
  --checkpointing_steps=2000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --target_idx="0,1,2,3,4,5,6,7,8,9" \
  --num_samples_per_target_idx=100 \
  --num_innocent_samples=100 \
  --output_dir="../sd15_target_relearned_0-9" \
```

After we obtain the releraned model, we can conduct mask learning via:
```
bash run_mask_learner.sh
```
Make sure to pass the correct parameters (e.g., freeze_ratio). Here is an example of `run_mask_learner.sh`:

```
accelerate launch mask_learner.py \
  --pretrained_model_name_or_path="../sd15_target_unlearned" \
  --finetuned_model_name_or_path="../sd15_target_relearned_0-9" \
  --target_dataset_name="../ff25" \
  --innocent_dataset_name="logo-wizard/modern-logo-dataset" \
  --caption_column="text" \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=16 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --max_train_steps=2000 \
  --checkpointing_steps=2000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --target_idx="0,1,2,3,4,5,6,7,8,9" \
  --num_samples_per_target_idx=100 \
  --num_innocent_samples=100 \
  --temperature=0.2 \
  --freeze_ratio=0.2 \
  --c_sparsity=5e3 \
  --mask_lr=1e1 \
  --user_lr=1e-5 \
  --user_interval=5 \
  --output_dir="aux-model" \
  --mask_output_dir="mask_t_0-9_i_logo_20_included" \
  --plot_mask \
```

### Apply the Learned Mask
On target person faces dataset:
```
bash run_target_trainer.sh
```
For example, to fine-tune on the target person with idx 0:
```
export MASK_DIR="mask_t_0-9_i_logo_20_included"

export MODEL_NAME="../sd15_target_unlearned"
export TRAIN_DIR="../ff25"
export OUTPUT_DIR="../sd15_target_relearn_0_m20"

accelerate launch target_trainer.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$TRAIN_DIR \
  --split="test" \
  --caption_column="text" \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --max_train_steps=2000 \
  --checkpointing_steps=2000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --selected_idx="0" \
  --num_samples_per_class=2000 \
  --output_dir=$OUTPUT_DIR \
  --finetune_scheme="learned_mask" \
  --random_mask_ratio=0.2 \
  --mask_dir=$MASK_DIR \
```

Similarly, on innocent dataset, such as logo and clothes datasets:
```
bash run_innocent_trainer.sh
```

### Evaluate Blocking Power
On target person faces dataset:
```
bash run_evaluate_target.sh
```
On innocent dataset:
```
bash run_evaluate_innocent.sh
```
You can customize the evaluation process, for example:
```
python3 evaluate_fid_topiq_target.py \
  --description="full fine-tuning vs. 20% freezing ratio on target 0" \
  --data_dir="../ff25" \
  --selected_idx=0 \
  --ft_model="../sd15_target_relearn_0_full" \
  --mft_model="../sd15_target_relearn_0_m20" \
  --num_samples=50 \
  --seed=6666 \
  --result_dir="sd15_t_0_full_vs_m20.txt" \
```
## Citation
```
@article{huang2024freezeasguard,
  title={FreezeAsGuard: Mitigating Illegal Adaptation of Diffusion Models via Selective Tensor Freezing},
  author={Huang, Kai and Gao, Wei},
  journal={arXiv preprint arXiv:2405.17472},
  year={2024}
}
```
