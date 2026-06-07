#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
cd imputation

# nohup ./TimesNet_retrieval.sh > ../logs/TimesNet_weather_retrieval.log 2>&1 &
model_name=TimesNet_retrieval
model_emb=TimesNet

mask_rates=(0.125 0.25 0.375 0.5) # 0.25 0.375 0.5
fuse_rates=(0.4)
top_ks=(3) # 3 5

for k in "${top_ks[@]}"
do
    for fuse in "${fuse_rates[@]}"
    do
        for rate in "${mask_rates[@]}"
        do
            echo "Running: Mask=$rate, Fuse=$fuse, Top_k=$k"

            python -u run.py \
                --task_name imputation_retrieval \
                --sheet_name 'polyencoder_freeze_backbone_retrieval_22_3' \
                --ablation_arch "freeze-backbone-retrieval + learnable fusing" \
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
                --d_model 64 \
                --d_ff 64 \
                --des 'Exp' \
                --itr 1 \
                --top_k 3 \
                --learning_rate 0.001 \
                --k $k \
                --fuse_rate $fuse \
                --representation_mode 'mean_pooling' \
                --retrieval_checkpoint_path "imputation/imputation_retriever/checkpoints_retriever/Transformer_weather_mask_${rate}_custom_ftM_sl96_ll0_pl0_dm128_nh8_el2_dl1_df128_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0/checkpoint.pth" \
                --checkpoints ./checkpoints_imputation_retrieval/
        done
    done
done
