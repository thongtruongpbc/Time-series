export CUDA_VISIBLE_DEVICES=0
cd /mnt/time-series/thongtx/imputation/polyencoder_retriever
model_name=Transformer
model_emb=Transformer

# nohup ./Transformer.sh > ../logs/Transformer_weather.log 2>&1 &
# delete job: pkill -9 -f run_retriever.py
for rate in 0.125 0.25 0.375 0.5 # 0.25 0.375 0.5
do
  echo "Running experiment with mask_rate: $rate"

  python -u run_retriever.py \
    --task_name retriever \
    --sheet_name 'polyencoder_retrieval' \
    --ablation_arch "polyencoder-retrieval" \
    --is_training 1 \
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
    --model_id "weather_mask_$rate" \
    --mask_rate $rate \
    --model $model_name \
    --model_emb $model_emb \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 0 \
    --pred_len 0 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --batch_size 16 \
    --d_model 128 \
    --d_ff 128 \
    --des 'Exp' \
    --itr 1 \
    --top_k 5 \
    --topm 10 \
    --learning_rate 0.0001 \
    --checkpoints ./checkpoints_retriever/
done
