export CUDA_VISIBLE_DEVICES=0
cd /mnt/time-series/time-series/thongtx/imputation
model_name=TimesNet
model_emb=TimesNet

# nohup ./TimesNet_ETT.sh > ../logs/TimesNet_ETT.log 2>&1 &



#etth2
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

#ettm2
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
      --root_path ./dataset/ETT-small/ \
      --data_path ETTm2.csv \
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
