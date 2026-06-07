export CUDA_VISIBLE_DEVICES=0
cd /mnt/time-series/time-series/thongtx/imputation
# nohup ./ModernTCN.sh > ../logs/ModernTCN_ECL.log 2>&1 &

THRESHOLD_VRAM=2000
THRESHOLD_CPU=80

check_resources() {
    while true; do
        FREE_VRAM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sed -n '1p')

        CPU_LOAD=$(uptime | awk -F'load average:' '{ print $2 }' | cut -d',' -f1 | xargs)
        CPU_LOAD_INT=$(printf "%.0f" $CPU_LOAD)

        echo "Checking: Free VRAM: ${FREE_VRAM}MB, CPU Load: ${CPU_LOAD}%"

        if [ "$FREE_VRAM" -gt "$THRESHOLD_VRAM" ] && [ "$CPU_LOAD_INT" -lt "$THRESHOLD_CPU" ]; then
            echo "Resources OK. Starting task..."
            break
        else
            echo "Resources busy (VRAM < $THRESHOLD_VRAM or CPU > $THRESHOLD_CPU). Waiting 15 mins..."
            sleep 900
        fi
    done
}

model_name=ModernTCN
model_emb=ModernTCN

for rate in 0.125 0.125 0.25 0.375 0.5 #0.125 0.25 0.375 0.5
do
  for len in 96 192 336 720 #192 336 720
  do
    if [ "$len" -eq 96 ] && [ "$rate" = "0.125" ]; then
      echo "Skipping experiment with mask_rate: $rate and len: $len"
      continue
    fi

    if [ "$len" -eq 192 ] && [ "$rate" = "0.125" ]; then
      echo "Skipping experiment with mask_rate: $rate and len: $len"
      continue
    fi

    echo "Running experiment with mask_rate: $rate"

    python -u run.py \
      --task_name imputation \
      --sheet_name 'Backbone_ICDM' \
      --ablation_arch 'baseline' \
      --is_training 1 \
      --root_path ./dataset/electricity/ \
      --data_path electricity.csv \
      --model_id "ECL_mask_$rate" \
      --mask_rate $rate \
      --model $model_name \
      --model_emb $model_emb \
      --data custom \
      --features M \
      --seq_len $len \
      --label_len 0 \
      --pred_len 0 \
      --e_layers 3 \
      --n_heads 16 \
      --dropout 0.2 \
      --patch_len 16 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 321 \
      --dec_in 321 \
      --c_out 321 \
      --batch_size 16 \
      --ffn_ratio 1 \
      --patch_size 1 \
      --patch_stride 1 \
      --num_blocks 1 \
      --large_size 71 \
      --small_size 5 \
      --dims 128 128 128 128 \
      --head_dropout 0.0 \
      --d_model 128 \
      --d_ff 256 \
      --des 'Exp' \
      --itr 1 \
      --top_k 5 \
      --learning_rate 0.001 \
      --train_epochs 100 \
      --patience 10 \
      --lradj type3 \
      --des Exp \
      --use_multi_scale False \
      --small_kernel_merged False \
      --representation_mode 'mean_pooling' \
      --checkpoints ./checkpoints_imputation/
  done
done
