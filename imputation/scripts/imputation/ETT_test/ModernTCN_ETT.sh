#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
cd /mnt/time-series/time-series/thongtx/imputation

model_name=ModernTCN
model_emb=ModernTCN
mask_rates=(0.125 0.25 0.375 0.5)
lens=(96 192 336 720)

#ETTm1
mask_rates=(0.375)
lens=(720)

for rate in "${mask_rates[@]}"
do
  for len in "${lens[@]}"
  do
    echo ">>>> Running ModernTCN: Mask=$rate, Seq_Len=$len <<<<"

    python -u run.py \
      --task_name imputation \
      --sheet_name 'Backbone_ICDM' \
      --ablation_arch 'baseline' \
      --is_training 1 \
      --root_path ./all_datasets/ETT-small/ \
      --data_path ETTm1.csv \
      --data ETTm1 \
      --model_id "ETTm1_mask_${rate}_sl${len}" \
      --model $model_name \
      --model_emb $model_emb \
      --mask_rate $rate \
      --features M \
      --seq_len $len \
      --label_len 0 \
      --pred_len 0 \
      --batch_size 16 \
      --learning_rate 0.001 \
      --train_epochs 100 \
      --patience 3 \
      --lradj type3 \
      --itr 1 \
      --des 'Exp' \
      --ffn_ratio 1 \
      --patch_size 1 \
      --patch_stride 1 \
      --num_blocks 1 \
      --large_size 71 \
      --small_size 5 \
      --dims 128 128 128 128 \
      --head_dropout 0.0 \
      --enc_in 7 \
      --dropout 0.0 \
      --use_multi_scale False \
      --small_kernel_merged False \
      --checkpoints ./checkpoints_imputation/
  done
done

mask_rates=(0.5)
lens=(192 336 720)

for rate in "${mask_rates[@]}"
do
  for len in "${lens[@]}"
  do
    echo ">>>> Running ModernTCN: Mask=$rate, Seq_Len=$len <<<<"

    python -u run.py \
      --task_name imputation \
      --sheet_name 'Backbone_ICDM' \
      --ablation_arch 'baseline' \
      --is_training 1 \
      --root_path ./all_datasets/ETT-small/ \
      --data_path ETTm1.csv \
      --data ETTm1 \
      --model_id "ETTm1_mask_${rate}_sl${len}" \
      --model $model_name \
      --model_emb $model_emb \
      --mask_rate $rate \
      --features M \
      --seq_len $len \
      --label_len 0 \
      --pred_len 0 \
      --batch_size 16 \
      --learning_rate 0.001 \
      --train_epochs 100 \
      --patience 3 \
      --lradj type3 \
      --itr 1 \
      --des 'Exp' \
      --ffn_ratio 1 \
      --patch_size 1 \
      --patch_stride 1 \
      --num_blocks 1 \
      --large_size 71 \
      --small_size 5 \
      --dims 128 128 128 128 \
      --head_dropout 0.0 \
      --enc_in 7 \
      --dropout 0.0 \
      --use_multi_scale False \
      --small_kernel_merged False \
      --checkpoints ./checkpoints_imputation/
  done
done


#ETThm2
mask_rates=(0.125 0.25 0.375 0.5)
lens=(96 192 336 720)

for rate in "${mask_rates[@]}"
do
  for len in "${lens[@]}"
  do
    echo ">>>> Running ModernTCN: Mask=$rate, Seq_Len=$len <<<<"

    python -u run.py \
      --task_name imputation \
      --sheet_name 'Backbone_ICDM' \
      --ablation_arch 'baseline' \
      --is_training 1 \
      --root_path ./all_datasets/ETT-small/ \
      --data_path ETTm2.csv \
      --data ETTm2 \
      --model_id "ETTm2_mask_${rate}_sl${len}" \
      --model $model_name \
      --model_emb $model_emb \
      --mask_rate $rate \
      --features M \
      --seq_len $len \
      --label_len 0 \
      --pred_len 0 \
      --batch_size 16 \
      --learning_rate 0.001 \
      --train_epochs 100 \
      --patience 3 \
      --lradj type3 \
      --itr 1 \
      --des 'Exp' \
      --ffn_ratio 1 \
      --patch_size 1 \
      --patch_stride 1 \
      --num_blocks 1 \
      --large_size 71 \
      --small_size 5 \
      --dims 128 128 128 128 \
      --head_dropout 0.0 \
      --enc_in 7 \
      --dropout 0.0 \
      --use_multi_scale False \
      --small_kernel_merged False \
      --checkpoints ./checkpoints_imputation/
  done
done
