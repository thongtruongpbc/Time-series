export CUDA_VISIBLE_DEVICES=0
cd /mnt/time-series/time-series/thongtx/imputation

# nohup ./RLinear.sh > ../logs/RLinear_ECL.log 2>&1 &

#RLinear

model_name=RLinear
model_emb=RLinear

for rate in 0.25 0.375 0.5 #0.125 0.25 0.375 0.5
do
  echo "Running experiment with mask_rate: $rate"
  for len in 720
  do
    if [ "$len" -eq 96 ]; then
      echo "Skipping experiment with mask_rate: $rate and len: $len"
      continue
    fi

    python -u run.py \
      --task_name imputation \
      --sheet_name 'Backbone_ICDM' \
      --ablation_arch 'baseline' \
      --is_training 0 \
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
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 321 \
      --dec_in 321 \
      --c_out 321 \
      --batch_size 16 \
      --des 'Exp' \
      --itr 1 \
      --top_k 5 \
      --learning_rate 0.001 \
      --representation_mode 'mean_pooling' \
      --checkpoints ./checkpoints_imputation/ \
      --dropout 0.0
  done
done


for rate in 0.5 #0.125 0.25 0.375 0.5
do
  echo "Running experiment with mask_rate: $rate"
  for len in 336
  do
    if [ "$len" -eq 96 ]; then
      echo "Skipping experiment with mask_rate: $rate and len: $len"
      continue
    fi

    python -u run.py \
      --task_name imputation \
      --sheet_name 'Backbone_ICDM' \
      --ablation_arch 'baseline' \
      --is_training 0 \
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
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 321 \
      --dec_in 321 \
      --c_out 321 \
      --batch_size 16 \
      --des 'Exp' \
      --itr 1 \
      --top_k 5 \
      --learning_rate 0.001 \
      --representation_mode 'mean_pooling' \
      --checkpoints ./checkpoints_imputation/ \
      --dropout 0.0
  done
done
