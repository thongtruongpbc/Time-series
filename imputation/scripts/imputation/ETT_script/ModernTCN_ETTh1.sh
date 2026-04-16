#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
cd /mnt/time-series/time-series/thongtx/imputation

model_name=ModernTCN
model_emb=ModernTCN
# nohup ./ModernTCN_ETTh1.sh > ../logs/ModernTCN_ETTh1.log 2>&1 &

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
      --data_path ETTh1.csv \
      --data ETTh1 \
      --model_id "ETTh1_mask_${rate}_sl${len}" \
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


# model_name=ModernTCN
# model_emb=ModernTCN_retrieval
# # nohup ./ModernTCN_ETTh1.sh > ../logs/ModernTCN_ETTh1.log 2>&1 &

# mask_rates=(0.125 0.25 0.375 0.5)
# lens=(96 192 336 720)

# for rate in "${mask_rates[@]}"
# do
#   for len in "${lens[@]}"
#   do
#     echo ">>>> Running ModernTCN: Mask=$rate, Seq_Len=$len <<<<"

#     python -u run.py \
#       --task_name imputation \
#       --sheet_name 'polyencoder_retrieval_ICDM' \
#       --ablation_arch "freeze-backbone-retrieval + learnable fusing" \
#       --is_training 1 \
#       --root_path ./all_datasets/ETT-small/ \
#       --data_path ETTh1.csv \
#       --data ETTh1 \
#       --model_id "ETTh1_mask_${rate}_sl${len}" \
#       --model $model_name \
#       --model_emb $model_emb \
#       --mask_rate $rate \
#       --features M \
#       --seq_len $len \
#       --label_len 0 \
#       --pred_len 0 \
#       --batch_size 16 \
#       --learning_rate 0.001 \
#       --train_epochs 100 \
#       --patience 3 \
#       --lradj type3 \
#       --itr 1 \
#       --des 'Exp' \
#       --ffn_ratio 1 \
#       --patch_size 1 \
#       --patch_stride 1 \
#       --num_blocks 1 \
#       --large_size 71 \
#       --small_size 5 \
#       --dims 128 128 128 128 \
#       --head_dropout 0.0 \
#       --enc_in 7 \
#       --dropout 0.0 \
#       --use_multi_scale False \
#       --small_kernel_merged False \
#       --representation_mode 'mean_pooling' \
#       --retrieval_checkpoint_path "/mnt/time-series/time-series/thongtx/imputation/polyencoder_retriever/checkpoints_retriever/Transformer_ETTh1_mask_${rate}_ETTh1_ftM_sl${len}_ll0_pl0_dm16_nh8_el2_dl1_df64_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0/checkpoint.pth" \
#       --checkpoints ./checkpoints_imputation_retrieval/
#   done
# done
