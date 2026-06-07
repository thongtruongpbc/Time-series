For download benchmark datasets, please to run:
python download_datasets.py

cài đặt dependencies:
pip install -r requirements.txt 
For LEA contrastive training:
cd imputation/retriever/scripts/retriever
cd ETT_script
./Transformer_ETTh1.sh
Tương tự với các case khác.

For retrieval-augmented time series imputation training:
first run baseline by:
cd imputation/scripts/imputation
cd ETT_script
./Autoformer_ETTh1.sh

Second run baseline + ALER-TI:
cd imputation/scripts/imputation
./Autoformer_ETTh1_retrieval.sh


