#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
cd /mnt/time-series/time-series/time-series/thongtx/imputation

# nohup ./Autoformer_ETTh1_retrieval.sh > ../logs/Autoformer_ETTh1_retrieval.log 2>&1 &

model_name=Autoformer_retrieval
model_emb=Autoformer

mask_rates=(0.125) # 0.25 0.375 0.5
fuse_rates=(1)
top_ks=(3) # 3 5
lens=(96 192 336 720)

for k in "${top_ks[@]}"
do
    for fuse in "${fuse_rates[@]}"
    do
        for rate in "${mask_rates[@]}"
        do
          for len in "${lens[@]}"
          do
            echo "Running: Mask=$rate, Fuse=$fuse, Top_k=$k"

            python -u run.py \
                --task_name imputation_retrieval \
                --sheet_name 'polyencoder_freeze_backbone_retrieval' \
                --ablation_arch "freeze-backbone-retrieval + learnable fusing" \
                --is_training 1 \
                --root_path ./dataset/ETT-small/ \
                --data_path ETTh1.csv \
                --model_id "ETTh1_mask_${rate}" \
                --mask_rate $rate \
                --model $model_name \
                --model_emb $model_emb \
                --data ETTh1 \
                --features M \
                --seq_len len \
                --label_len 0 \
                --pred_len 0 \
                --e_layers 2 \
                --d_layers 1 \
                --factor 3 \
                --enc_in 7 \
                --dec_in 7 \
                --c_out 7 \
                --batch_size 32 \
                --d_model 128 \
                --d_ff 512 \
                --des 'Exp' \
                --itr 1 \
                --top_k $k \
                --fuse_rate $fuse \
                --learning_rate 0.001 \
                --representation_mode 'mean_pooling' \
                --retrieval_checkpoint_path "imputation/imputation_retriever/checkpoints_retriever_tmp/Transformer_ETTh1_mask_${rate}_ETTh1_ftM_sl96_ll0_pl0_dm16_nh8_el2_dl1_df64_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0/checkpoint.pth" \
                --train_epochs 1 \
                --checkpoints ./checkpoints_imputation_retrieval/
          done
        done
    done
done
