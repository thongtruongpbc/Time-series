# Selective Learning for Deep Time Series Forecasting

<div align="center">

[![BasicTS](https://img.shields.io/badge/Developing%20with-BasicTS-2077ff.svg)](https://github.com/GestaltCogTeam/BasicTS)
[![arXiv](https://img.shields.io/badge/arXiv-2510.25207-b31b1b.svg)](https://arxiv.org/abs/2510.25207)

</div>

<div align="center">

[**English**](./README.md) **|**
[**简体中文**](./README_CN.md)

**Official implementation of the paper [Selective Learning for Deep Time Series Forecasting](https://arxiv.org/abs/2510.25207).**

</div>

![img.png](images/framework.png)

## 📌 Abstract

> **Selective Learning** is a novel and powerful training strategy designed to make deep time series forecasting (TSF) models more robust against overfitting. Instead of uniformly learning from all timesteps, it selectively focuses on reliable ones while filtering out uncertain or anomalous samples. This is achieved through two complementary masks — an uncertainty mask based on residual entropy and an anomaly mask using residual lower-bound estimation. Experiments on eight benchmark datasets show that Selective Learning consistently improves performance, reducing MSE by 37.4% on Informer, 8.4% on TimesNet, and 6.5% on iTransformer.

## 🛠️ Usage

1. **Environment and Dependencies**  
   This project is built upon the [BasicTS](https://github.com/GestaltCogTeam/BasicTS) time series benchmarking library.  
   Download the package with version ≥ 1.0.0 from [this release link](https://github.com/GestaltCogTeam/BasicTS/releases/tag/v1.0.0) and install it with pip.
   ```bash
	pip install basicts-1.0-py3-none-any.whl
	```
   (or pip install -r requirments.txt)

2. **Prepare Datasets**  
   All datasets used in the paper are natively supported in BasicTS.  
   Please refer to the [dataset documentation](https://github.com/GestaltCogTeam/BasicTS/blob/master/docs/dataset_design_cn.md) for instructions on downloading and using them.  

3. **Train the Estimation Model**  
   Each dataset requires training an estimation model first to guide anomaly masking.  
   For example, you can train a DLinear estimation model on the ETTh1 dataset as shown in `train_est_model` of the [demo](demo.py), and the logs and checkpoints will be saved under `project_root_path/checkpoints/DLinear/`.

4. **Train the Main Model with Selective Learning**
      ```bash
	python demo.py
	```

   BasicTS now natively supports **Selective Learning**.  
   You can simply add the `SelectiveLearning` callback in your configuration to enable selective learning during training.
   ```python
	from basicts.runners.callback import SelectiveLearning
	
	estimator = DLinear(DLinearConfig(...))
	
	sl_callback = SelectiveLearning(
		r_u=0.3, # uncertainty masking ratio
		r_a=0.3, # anomaly masking ratio
		estimator=estimator, # estimation model
		ckpt_path="checkpoints/DLinear/..." # .pt file
	)
	
	config = BasicTSForecastingConfig(
		#..., your other config
		callbacks=[sl_callback] # add selective learning callback
	)
	```

   The function `train_main_model` in [demo](demo.py) shows how to train iTransformer with selective learning.

## 📈 Results

![img.png](images/main_result.png)

## 🔗 Citation

🔥🔥🔥 **If you find this repository useful, please consider citing our NeurIPS'25 paper!** 🔥🔥🔥

```tex
@misc{fu2025selectivelearningdeeptime,
      title={Selective Learning for Deep Time Series Forecasting}, 
      author={Yisong Fu and Zezhi Shao and Chengqing Yu and Yujie Li and Zhulin An and Qi Wang and Yongjun Xu and Fei Wang},
      year={2025},
      eprint={2510.25207},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.25207}, 
}
```
