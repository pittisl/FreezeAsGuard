export MASK_DIR="mask_t_0_4_i_logo_20_included"

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
  --checkpointing_steps=4000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --selected_idx="0" \
  --num_samples_per_class=2000 \
  --output_dir=$OUTPUT_DIR \
  --finetune_scheme="learned_mask" \
  --mask_dir=$MASK_DIR \


export MODEL_NAME="../sd15_target_unlearned"
export TRAIN_DIR="../ff25"
export OUTPUT_DIR="../sd15_target_relearn_4_m20"

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
  --checkpointing_steps=4000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --selected_idx="4" \
  --num_samples_per_class=2000 \
  --output_dir=$OUTPUT_DIR \
  --finetune_scheme="learned_mask" \
  --mask_dir=$MASK_DIR \