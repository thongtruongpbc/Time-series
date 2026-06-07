# ALER-TI: Aligned Latent Embedding Retrieval for Time-series Imputation

## Overview

**ALER-TI** is a retrieval-augmented framework for time series imputation that explicitly leverages historical patterns to supplement degraded local context. The core component, **Latent Embedding Alignment (LEA)**, mitigates the representation mismatch between corrupted queries and complete historical candidates via post-hoc latent-space masking — enabling pre-computed candidate caching while preserving query-aware alignment.

ALER-TI is **model-agnostic** and can be integrated with any imputation backbone through a lightweight adapter without modifying its internal design.

<p align="center">
  <img src="figures/overview.png" width="800"/>
</p>

---

## Requirements

```bash
pip install -r requirements.txt
```

---

## Datasets

```bash
python download_datasets.py
```

Datasets used: **ETTh1, ETTh2, ETTm1, ETTm2, Electricity, Weather**.

---

## Usage

### Stage 1 — LEA Contrastive Training

Train the retriever on each dataset/backbone combination:

```bash
cd imputation/retriever/scripts/retriever/ETT_script
./Transformer_ETTh1.sh
```

Repeat for other datasets and configurations under `ETT_script/` and equivalent directories.

### Stage 2 — Retrieval-Augmented Imputation Training

**Step 1 — Train the backbone baseline:**

```bash
cd imputation/scripts/imputation/ETT_script
./Autoformer_ETTh1.sh
```

**Step 2 — Train backbone + ALER-TI:**

```bash
cd imputation/scripts/imputation
./Autoformer_ETTh1_retrieval.sh
```

---

## Main Results

### MSE Improvement over Baselines

ALER-TI consistently improves **10 backbone models** across **6 datasets** and **4 missing rates** $r \in \{0.125, 0.25, 0.375, 0.5\}$.

<p align="center">
  <img src="promotion.png" width="900"/>
</p>


### Detailed MSE Results

Full imputation MSE results across all datasets, missing rates, and sequence lengths $L \in \{96, 192, 336, 720\}$:

<p align="center">
  <img src="detailed_mse.png" width="900"/>
</p>

---


```
