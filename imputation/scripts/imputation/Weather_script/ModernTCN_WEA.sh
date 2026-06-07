export CUDA_VISIBLE_DEVICES=0
cd /mnt/time-series/time-series/thongtx/imputation
model_name=ModernTCN
model_emb=ModernTCN

# nohup ./ModernTCN_WEA.sh > ../logs/ModernTCN_WEA.log 2>&1 &

for rate in 0.125 0.25 0.375 0.5
do
  for len in 96 192 336 720
  do
    echo "Running experiment with mask_rate: $rate"
    python -u run.py \
      --task_name imputation \
      --sheet_name 'Backbone_ICDM' \
      --ablation_arch 'baseline' \
      --is_training 1 \
      --root_path ./dataset/weather/ \
      --data_path weather.csv \
      --model_id "weather_mask_$rate" \
      --mask_rate $rate \
      --model $model_name \
      --model_emb $model_emb \
      --data custom \
      --features M \
      --seq_len $len \
      --label_len 0 \
      --pred_len 0 \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 21 \
      --dec_in 21 \
      --c_out 21 \
      --batch_size 16 \
      --ffn_ratio 1 \
      --patch_size 1 \
      --patch_stride 1 \
      --num_blocks 1 \
      --large_size 71 \
      --small_size 5 \
      --dims 128 128 128 128 \
      --head_dropout 0.0 \
      --dropout 0.1 \
      --use_multi_scale False \
      --small_kernel_merged False \
      --representation_mode 'mean_pooling' \
      --checkpoints ./checkpoints_imputation/
  done
done
