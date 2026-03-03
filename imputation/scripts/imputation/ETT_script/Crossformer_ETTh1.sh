export CUDA_VISIBLE_DEVICES=0
cd /mnt/time-series/thongtx/imputation
model_name=Crossformer
model_emb=Crossformer

for rate in 0.125 0.25 0.375 0.5
do
  echo "Running experiment with mask_rate: $rate"

  python -u run.py \
    --task_name imputation \
    --sheet_name 'freeze_backbone_retrieval' \
    --ablation_arch 'baseline' \
    --is_training 1 \
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
    --d_model 64 \
    --d_ff 64 \
    --des 'Exp' \
    --itr 1 \
    --top_k 5 \
    --learning_rate 0.001 \
    --representation_mode 'mean_pooling' \
    --checkpoints ./checkpoints_imputation/
done
