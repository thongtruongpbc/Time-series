export CUDA_VISIBLE_DEVICES=0
cd /mnt/time-series/time-series/thongtx/imputation
model_name=TimeBridge
model_emb=TimeBridge

for rate in 0.125 0.25 0.375 0.5
do
  echo "Running experiment with mask_rate: $rate"

  python -u run.py \
    --task_name imputation \
    --sheet_name 'Backbone_ICDM' \
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
    --ca_layers 1 \
    --pd_layers 1 \
    --ia_layers 1 \
    --des 'Exp' \
    --period 48 \
    --num_p 12 \
    --d_model 128 \
    --d_ff 128 \
    --batch_size 32 \
    --alpha 0.35 \
    --learning_rate 0.0002 \
    --train_epochs 1 \
    --patience 3 \
    --representation_mode 'mean_pooling' \
    --checkpoints ./checkpoints_imputation/
done
