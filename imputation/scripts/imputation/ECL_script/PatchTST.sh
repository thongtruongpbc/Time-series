export CUDA_VISIBLE_DEVICES=0
cd /mnt/time-series/time-series/thongtx/imputation
# nohup ./PatchTST.sh > ../logs/PatchTST_ECL.log 2>&1 &

model_name=PatchTST
model_emb=PatchTST

for rate in 0.125 0.25 0.375 0.5 #0.125 0.25 0.375 0.5
do
  for len in 96 192 336 720
  do
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
      --batch_size 32 \
      --d_model 128 \
      --d_ff 256 \
      --des 'Exp' \
      --itr 1 \
      --top_k 5 \
      --learning_rate 0.001 \
      --train_epochs 100 \
      --representation_mode 'mean_pooling' \
      --checkpoints ./checkpoints_imputation/
  done
done


#Crossformer

model_name=Crossformer
model_emb=Crossformer

for rate in 0.125 0.25 0.375 0.5 #0.125 0.25 0.375 0.5
do
  echo "Running experiment with mask_rate: $rate"
  for len in 96 192 336 720
  do
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
      --e_layers 2 \
      --d_layers 1 \
      --n_heads 2 \
      --factor 3 \
      --enc_in 321 \
      --dec_in 321 \
      --c_out 321 \
      --batch_size 16 \
      --d_model 64 \
      --d_ff 128 \
      --des 'Exp' \
      --itr 1 \
      --top_k 5 \
      --learning_rate 5e-4 \
      --representation_mode 'mean_pooling' \
      --checkpoints ./checkpoints_imputation/
  done
done
