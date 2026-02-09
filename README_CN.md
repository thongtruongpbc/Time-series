# Selective Learning for Deep Time Series Forecasting

<div align="center">

[![BasicTS](https://img.shields.io/badge/Developing%20with-BasicTS-2077ff.svg)](https://github.com/GestaltCogTeam/BasicTS)
[![arXiv](https://img.shields.io/badge/arXiv-2510.25207-b31b1b.svg)](https://arxiv.org/abs/2510.25207)

</div>

<div align="center">

[**English**](./README.md) **|**
[**简体中文**](./README_CN.md)

**论文[Selective Learning for Deep Time Series Forecasting](https://arxiv.org/abs/2510.25207)的官方代码实现。**

</div>

![img.png](images/framework.png)

## 📌 摘要

> **选择学习**是一种新颖且强大的深度时间序列预测训练策略，能够增强模型的泛化性以及对过拟合的鲁棒性。不同于传统方法对所有时间步一视同仁地学习，它通过筛选可靠时间步、过滤不确定和异常样本来提高模型的泛化能力。该方法基于两种互补的掩码机制：利用残差熵的不确定性掩码与基于残差下界估计的异常掩码。在八个真实世界数据集上的实验结果表明，选择学习显著提升了多种主流模型的预测性能，在 Informer、TimesNet 和 iTransformer 上分别带来了 37.4%、8.4% 和 6.5% 的 MSE 降低。

## 🛠️ 使用方法

1. **环境与依赖**
	本项目基于时序分析工具评测库[BasicTS](https://github.com/GestaltCogTeam/BasicTS)开发。在[此链接](https://github.com/GestaltCogTeam/BasicTS/releases/tag/v1.0.0)下载版本号>=1.0.0的安装包后，在终端执行：
	```bash
	pip install basicts-1.0-py3-none-any.whl
	```
2. **准备数据集**
	论文中所使用的数据集BasicTS均内置支持，请参考BasicTS的[数据集文档](https://github.com/GestaltCogTeam/BasicTS/blob/master/docs/dataset_design_cn.md)下载。
3. **训练估计模型**
	对每个数据集需要先训练估计模型以指导异常掩码。[demo](demo.py)中的`train_est_model`展示了在ETTh1数据集上训练DLinear作为估计模型。
	训练日志和模型权重会自动保存在`project_root_path/checkpoints/DLinear/`路径下。
4. **使用选择学习训练主模型**
	BasicTS现已内置支持选择学习，在配置中添加`SelectiveLearning`回调即可使用选择学习进行训练。
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

	[demo](demo.py)中的`train_main_model`展示了如何使用选择学习训练iTransformer。

## 📈 结果

![img.png](images/main_result.png)

## 🔗 引用

🔥🔥🔥 **如果您觉得该仓库有帮助的话，请考虑引用我们NeurIPS'25的论文** 🔥🔥🔥

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
