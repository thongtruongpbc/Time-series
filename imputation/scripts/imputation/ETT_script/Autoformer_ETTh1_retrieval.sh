export CUDA_VISIBLE_DEVICES=0
cd /mnt/time-series/thongtx/imputation
model_emb=Transformer
model_name=Autoformer_retrieval
# 0.25 0.375 0.5
for rate in 0.125 0.25 0.375 0.5
do
  for ablation_arch in "Linear fuse" "Linear fuse + Cross-attention" "Samplewise-attention"
  do
    echo "Running experiment with mask_rate: $rate"
    python -u run.py \
      --task_name imputation_retrieval \
      --sheet_name 'backbone_retrieval' \
      --ablation_arch 'Linear fuse' \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTh1.csv \
      --model_id "ETTh1_mask_$rate" \
      --mask_rate $rate \
      --model $model_name \
      --data ETTh1 \
      --model_emb $model_emb \
      --features M \
      --seq_len 96 \
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
      --top_k 1 \
      --learning_rate 0.001 \
      --patience 5 \
      --representation_mode 'cls_token' \
      --checkpoints ./checkpoints_imputation_retrieval/
  done
done
