export CUDA_VISIBLE_DEVICES=0
cd /mnt/time-series/time-series/thongtx/imputation
# nohup ./MTSMixer_ETT.sh > ../logs/MTSMixer_ETT.log 2>&1 &

model_name=MTSMixer
model_emb=MTSMixer

for rate in 0.125 # 0.25 0.375 0.5 #0.125 0.25 0.375 0.5
do
  for len in 96 # 192 336 720
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
      --seq_len $len \
      --label_len 0 \
      --pred_len 0 \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --batch_size 32 \
      --d_model 64 \
      --d_ff 4 \
      --des 'Exp' \
      --itr 1 \
      --top_k 5 \
      --learning_rate 0.001 \
      --train_epochs 100 \
      --representation_mode 'mean_pooling' \
      --checkpoints ./checkpoints_imputation/
  done
done


# for rate in 0.125 0.25 0.375 0.5 #0.125 0.25 0.375 0.5
# do
#   for len in 96 192 336 720
#   do
#     echo "Running experiment with mask_rate: $rate"

#     python -u run.py \
#       --task_name imputation \
#       --sheet_name 'Backbone_ICDM' \
#       --ablation_arch 'baseline' \
#       --is_training 1 \
#       --root_path ./dataset/ETT-small/ \
#       --data_path ETTh2.csv \
#       --model_id "ETTh2_mask_$rate" \
#       --mask_rate $rate \
#       --model $model_name \
#       --model_emb $model_emb \
#       --data ETTh2 \
#       --features M \
#       --seq_len $len \
#       --label_len 0 \
#       --pred_len 0 \
#       --e_layers 2 \
#       --d_layers 1 \
#       --factor 3 \
#       --enc_in 7 \
#       --dec_in 7 \
#       --c_out 7 \
#       --batch_size 32 \
#       --d_model 512 \
#       --d_ff 2048 \
#       --des 'Exp' \
#       --itr 1 \
#       --top_k 5 \
#       --learning_rate 0.001 \
#       --train_epochs 100 \
#       --representation_mode 'mean_pooling' \
#       --checkpoints ./checkpoints_imputation/
#   done
# done


# SCINet
# model_name=SCINet
# model_emb=SCINet

# for rate in 0.125 0.25 0.375 0.5 #0.125 0.25 0.375 0.5
# do
#   for len in 96 192 336 720
#   do
#     echo "Running experiment with mask_rate: $rate"

#     python -u run.py \
#       --task_name imputation \
#       --sheet_name 'Backbone_ICDM' \
#       --ablation_arch 'baseline' \
#       --is_training 1 \
#       --root_path ./dataset/electricity/ \
#       --data_path electricity.csv \
#       --model_id "ECL_mask_$rate" \
#       --mask_rate $rate \
#       --model $model_name \
#       --model_emb $model_emb \
#       --data custom \
#       --features M \
#       --seq_len $len \
#       --label_len 0 \
#       --pred_len 0 \
#       --e_layers 2 \
#       --d_layers 1 \
#       --factor 3 \
#       --enc_in 321 \
#       --dec_in 321 \
#       --c_out 321 \
#       --batch_size 16 \
#       --d_model 64 \
#       --d_ff 128 \
#       --des 'Exp' \
#       --itr 1 \
#       --top_k 5 \
#       --learning_rate 0.001 \
#       --train_epochs 100 \
#       --representation_mode 'mean_pooling' \
#       --checkpoints ./checkpoints_imputation/
#   done
# done
