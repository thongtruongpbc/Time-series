#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
cd imputation
# nohup ./Autoformer_ETTm1_retrieval.sh >> ../logs/Autoformer_ETTm1_retrieval.log 2>&1 &
model_name=Autoformer_retrieval
model_emb=Autoformer

mask_rates=(0.125 0.25 0.375 0.5) # 0.25 0.375 0.5
fuse_rates=(1)
top_ks=(1 3 5) # 3 5

for rate in "${mask_rates[@]}"
do
    for fuse in "${fuse_rates[@]}"
    do
        model_name=Autoformer
        echo "Running baseline experiment with mask_rate: $rate"

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
            --seq_len 96 \
            --label_len 0 \
            --pred_len 0 \
            --e_layers 2 \
            --d_layers 1 \
            --factor 3 \
            --enc_in 7 \
            --dec_in 7 \
            --c_out 7 \
            --batch_size 16 \
            --d_model 128 \
            --d_ff 128 \
            --des 'Exp' \
            --itr 1 \
            --top_k 5 \
            --learning_rate 0.001 \
            --representation_mode 'mean_pooling' \
            --checkpoints ./checkpoints_imputation/

        model_name=Autoformer_retrieval
        for k in "${top_ks[@]}"
        do
            echo "Running: Mask=$rate, Fuse=$fuse, Top_k=$k"

            python -u run.py \
                --task_name imputation_retrieval \
                --sheet_name 'Backbone_ICDM' \
                --ablation_arch "freeze-backbone-retrieval + learnable fusing" \
                --is_training 1 \
                --root_path ./dataset/ETT-small/ \
                --data_path ETTm1.csv \
                --model_id "ETTm1_mask_${rate}" \
                --mask_rate $rate \
                --model $model_name \
                --model_emb $model_emb \
                --data ETTm1 \
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
                --batch_size 16 \
                --d_model 128 \
                --d_ff 128 \
                --des 'Exp' \
                --itr 1 \
                --top_k $k \
                --fuse_rate $fuse \
                --learning_rate 0.001 \
                --representation_mode 'mean_pooling' \
                --checkpoints ./checkpoints_imputation_retrieval/
        done
    done
done
