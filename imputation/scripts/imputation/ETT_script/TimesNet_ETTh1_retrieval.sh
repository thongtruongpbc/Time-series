export CUDA_VISIBLE_DEVICES=0
cd /mnt/time-series/thongtx/imputation
model_name=TimesNet_retrieval
model_emb=TimesNet
#  0.25 0.375 0.5
for rate in 0.125
do
  echo "Running experiment with mask_rate: $rate"

  python -u run.py \
    --task_name imputation_retrieval \
    --sheet_name 'backbone_retrieval' \
    --ablation_arch '(Token embedding) Two-branch backbone + Cross-attention' \
    --is_training 0 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id "ETTh1_mask_$rate" \
    --mask_rate $rate \
    --model $model_name \
    --model_emb $model_emb \
    --data ETTh1 \
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
    --d_model 16 \
    --d_ff 32 \
    --des 'Exp' \
    --itr 1 \
    --top_k 3 \
    --learning_rate 0.001 \
    --representation_mode 'mean_pooling' \
    --checkpoints ./checkpoints_imputation/
done
