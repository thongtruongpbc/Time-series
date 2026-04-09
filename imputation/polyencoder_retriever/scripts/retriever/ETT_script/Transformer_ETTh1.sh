export CUDA_VISIBLE_DEVICES=0
cd /mnt/time-series/time-series/thongtx/imputation/polyencoder_retriever
model_name=Transformer
model_emb=Transformer

# nohup ./Transformer_ETTh1.sh > ../logs/Transformer_ETTh1.log 2>&1 &
# delete job: pkill -9 -f run_retriever.py
for rate in 0.125 0.25 0.375 0.5 # 0.25 0.375 0.5
do
  for len in 96 192 336 720
  do
    echo "Running experiment with mask_rate: $rate"

    python -u run_retriever.py \
      --task_name retriever \
      --sheet_name 'polyencoder_retrieval' \
      --ablation_arch "polyencoder-retrieval" \
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
      --batch_size 8 \
      --d_model 16 \
      --d_ff 64 \
      --des 'Exp' \
      --itr 1 \
      --top_k 5 \
      --learning_rate 0.0001 \
      --checkpoints ./checkpoints_retriever/
  done
done

# python -u run.py \
#   --task_name imputation \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_mask_0.25 \
#   --mask_rate 0.25 \
#   --model $model_name \
#   --data ETTh1 \
#   --features M \
#   --seq_len 96 \
#   --label_len 0 \
#   --pred_len 0 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --batch_size 16 \
#   --d_model 128 \
#   --d_ff 128 \
#   --des 'Exp' \
#   --itr 1 \
#   --top_k 5 \
#   --learning_rate 0.001

# python -u run.py \
#   --task_name imputation \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_mask_0.375 \
#   --mask_rate 0.375 \
#   --model $model_name \
#   --data ETTh1 \
#   --features M \
#   --seq_len 96 \
#   --label_len 0 \
#   --pred_len 0 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --batch_size 16 \
#   --d_model 128 \
#   --d_ff 128 \
#   --des 'Exp' \
#   --itr 1 \
#   --top_k 5 \
#   --learning_rate 0.001

# python -u run.py \
#   --task_name imputation \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_mask_0.5 \
#   --mask_rate 0.5 \
#   --model $model_name \
#   --data ETTh1 \
#   --features M \
#   --seq_len 96 \
#   --label_len 0 \
#   --pred_len 0 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --batch_size 16 \
#   --d_model 128 \
#   --d_ff 128 \
#   --des 'Exp' \
#   --itr 1 \
#   --top_k 5 \
#   --learning_rate 0.001
