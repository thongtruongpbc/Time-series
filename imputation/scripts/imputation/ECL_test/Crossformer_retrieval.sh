#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
cd imputation
RUN_LIMIT="systemd-run --scope -p CPUQuota=90% -p CPUSchedulingPolicy=idle"

# nohup ./Crossformer_retrieval.sh >> ../logs/Crossformer_retrieval.log 2>&1 &
model_name=Crossformer_retrieval
model_emb=Crossformer

mask_rates=(0.125) # 0.25 0.375 0.5
fuse_rates=(1)
top_ks=(3) # 3 5
lens=(96)

for rate in "${mask_rates[@]}"
do
    for fuse in "${fuse_rates[@]}"
    do

        for k in "${top_ks[@]}"
        do
            for len in "${lens[@]}"
            do
                echo "Running: Mask=$rate, Fuse=$fuse, Top_k=$k"

                python -u run.py \
                --task_name imputation_retrieval \
                --sheet_name 'polyencoder_freeze_backbone_retrieval_29_3' \
                --ablation_arch "freeze-backbone-retrieval + learnable fusing" \
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
                --top_k $k \
                --fuse_rate $fuse \
                --learning_rate 0.001 \
                --representation_mode 'mean_pooling' \
                --retrieval_checkpoint_path "imputation/imputation_retriever/checkpoints_retriever/Transformer_ECL_mask_${rate}_custom_ftM_sl${len}_ll0_pl0_dm16_nh8_el2_dl1_df64_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0/checkpoint.pth" \
                --checkpoints ./checkpoints_imputation_retrieval/
            done
        done
    done
done
