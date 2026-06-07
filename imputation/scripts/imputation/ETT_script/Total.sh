#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
cd imputation
# nohup ./Total.sh > ../logs/Total.log 2>&1 &

#####etth1

#####etth2
model_name=ModernTCN
model_emb=ModernTCN
mask_rates=(0.5)
lens=(96)

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
      --data_path ETTh2.csv \
      --data ETTh2 \
      --model_id "ETTh2_mask_${rate}_sl${len}" \
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
      --dims 64 64 64 64 \
      --head_dropout 0.0 \
      --enc_in 7 \
      --dropout 0.0 \
      --use_multi_scale False \
      --small_kernel_merged False \
      --checkpoints ./checkpoints_imputation/
  done
done


model_name=ModernTCN
model_emb=ModernTCN
mask_rates=(0.375 0.5)
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
      --data_path ETTh2.csv \
      --data ETTh2 \
      --model_id "ETTh2_mask_${rate}_sl${len}" \
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
      --dims 64 64 64 64 \
      --head_dropout 0.0 \
      --enc_in 7 \
      --dropout 0.0 \
      --use_multi_scale False \
      --small_kernel_merged False \
      --checkpoints ./checkpoints_imputation/
  done
done


model_name=TimesNet
model_emb=TimesNet

for rate in 0.125
do
  for len in 336 720
  do
    echo "Running experiment with mask_rate: $rate"
    python -u run.py \
      --task_name imputation \
      --sheet_name 'Backbone_ICDM' \
      --ablation_arch 'baseline' \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTh2.csv \
      --model_id "ETTh2_mask_$rate" \
      --mask_rate $rate \
      --model $model_name \
      --model_emb $model_emb \
      --data ETTh2 \
      --features M \
      --seq_len $len \
      --label_len 0 \
      --pred_len 0 \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --batch_size 16 \
      --d_model 64 \
      --d_ff 128 \
      --des 'Exp' \
      --itr 1 \
      --top_k 3 \
      --learning_rate 0.001 \
      --representation_mode 'mean_pooling' \
      --checkpoints ./checkpoints_imputation/
  done
done

for rate in 0.25 0.375 0.5
do
  for len in 96 192 336 720
  do
    echo "Running experiment with mask_rate: $rate"
    python -u run.py \
      --task_name imputation \
      --sheet_name 'Backbone_ICDM' \
      --ablation_arch 'baseline' \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTh2.csv \
      --model_id "ETTh2_mask_$rate" \
      --mask_rate $rate \
      --model $model_name \
      --model_emb $model_emb \
      --data ETTh2 \
      --features M \
      --seq_len $len \
      --label_len 0 \
      --pred_len 0 \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --batch_size 16 \
      --d_model 64 \
      --d_ff 128 \
      --des 'Exp' \
      --itr 1 \
      --top_k 3 \
      --learning_rate 0.001 \
      --representation_mode 'mean_pooling' \
      --checkpoints ./checkpoints_imputation/
  done
done




#ettm1

model_name=Autoformer
model_emb=Autoformer

for rate in 0.125
do
  for len in 96
  do
    echo "Running experiment with mask_rate: $rate"
    python -u run.py \
      --task_name imputation \
      --sheet_name 'Backbone_ICDM' \
      --ablation_arch 'baseline' \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTm1.csv \
      --model_id "ETTm1_mask_$rate" \
      --mask_rate $rate \
      --model $model_name \
      --model_emb $model_emb \
      --data ETTm1 \
      --features M \
      --seq_len $len \
      --label_len 0 \
      --pred_len 0 \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --batch_size 16 \
      --d_model 128 \
      --d_ff 128 \
      --des 'Exp' \
      --itr 1 \
      --top_k 5 \
      --learning_rate 0.001 \
      --representation_mode 'mean_pooling' \
      --checkpoints ./checkpoints_imputation/
  done
done

for rate in 0.125 0.25 0.375 0.5
do
  for len in 192 336 720
  do
    echo "Running experiment with mask_rate: $rate"
    python -u run.py \
      --task_name imputation \
      --sheet_name 'Backbone_ICDM' \
      --ablation_arch 'baseline' \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTm1.csv \
      --model_id "ETTm1_mask_$rate" \
      --mask_rate $rate \
      --model $model_name \
      --model_emb $model_emb \
      --data ETTm1 \
      --features M \
      --seq_len $len \
      --label_len 0 \
      --pred_len 0 \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --batch_size 16 \
      --d_model 128 \
      --d_ff 128 \
      --des 'Exp' \
      --itr 1 \
      --top_k 5 \
      --learning_rate 0.001 \
      --representation_mode 'mean_pooling' \
      --checkpoints ./checkpoints_imputation/
  done
done

#ettm2
model_name=Autoformer
model_emb=Autoformer

for rate in 0.375 0.5
do
  for len in 96 192 336 720
  do
    echo "Running experiment with mask_rate: $rate"
    python -u run.py \
      --task_name imputation \
      --sheet_name 'Backbone_ICDM' \
      --ablation_arch 'baseline' \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTh2.csv \
      --model_id "ETTm2_mask_$rate" \
      --mask_rate $rate \
      --model $model_name \
      --model_emb $model_emb \
      --data ETTm2 \
      --features M \
      --seq_len $len \
      --label_len 0 \
      --pred_len 0 \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --batch_size 16 \
      --d_model 128 \
      --d_ff 128 \
      --des 'Exp' \
      --itr 1 \
      --top_k 5 \
      --learning_rate 0.001 \
      --representation_mode 'mean_pooling' \
      --checkpoints ./checkpoints_imputation/
  done
done


model_name=ModernTCN
model_emb=ModernTCN
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
