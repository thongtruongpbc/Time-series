export CUDA_VISIBLE_DEVICES=0
cd /mnt/time-series/time-series/thongtx/imputation/retriever
model_name=Transformer
model_emb=Transformer

# nohup ./Transformer.sh > ../logs/Transformer_electricity.log 2>&1 &
# delete job: pkill -9 -f run_retriever.py
for rate in 0.125 0.25 0.375 0.5 # 0.25 0.375 0.5
do
  echo "Running experiment with mask_rate: $rate"

  python -u run_retriever.py \
    --task_name retriever \
    --sheet_name 'polyencoder_retrieval' \
    --ablation_arch "polyencoder-retrieval" \
    --is_training 1 \
    --root_path ./dataset/electricity/ \
    --data_path electricity.csv \
    --model_id "ECL_mask_$rate" \
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
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --batch_size 16 \
    --d_model 16 \
    --d_ff 64 \
    --des 'Exp' \
    --itr 1 \
    --top_k 5 \
    --learning_rate 0.0001 \
    --checkpoints ./checkpoints_retriever/
done
