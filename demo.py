from basicts import BasicTSLauncher
from basicts.configs import BasicTSForecastingConfig
from basicts.models.DLinear import DLinear, DLinearConfig

from basicts.models.iTransformer import iTransformerForForecasting, iTransformerConfig
from basicts.runners.callback import SelectiveLearning


def train_est_model():
    
    for output_len in [96, 192, 336, 720]:
        
        input_len = 336
        num_epochs = 100

        model_config = DLinearConfig(
            input_len=input_len,
            output_len=output_len
        )

        config = BasicTSForecastingConfig(
            model=DLinear,
            model_config=model_config,
            dataset_name="ETTh1",
            input_len=input_len,
            output_len=output_len,
            batch_size=64,
            learning_rate=5e-4,
            num_epochs=num_epochs,
            gpus="0"
        )

        BasicTSLauncher.launch_training(config)

def train_main_model(checkpoint_paths: dict):

    for output_len in [96, 192, 336, 720]:
        
        input_len = 336
        num_epochs = 100

        model_config = iTransformerConfig(
            input_len=input_len,
            output_len=output_len,
            num_features=7
        )
        estimator = DLinear
        estimator_config = DLinearConfig(
            input_len=input_len,
            output_len=output_len,
            num_features=7
        )
        # estimator = DLinear(DLinearConfig(
        #     input_len=input_len, output_len=output_len))
        
        sl_callback = SelectiveLearning(
		    r_u=0.3, # uncertainty masking ratio
		    r_a=0.3, # anomaly masking ratio
		    estimator=estimator, # estimation model
            estimator_config=estimator_config,
		    ckpt_path=checkpoint_paths[output_len] # .pt file
	    )

        config = BasicTSForecastingConfig(
            model=iTransformerForForecasting,
            model_config=model_config,
            dataset_name="ETTh1",
            input_len=input_len,
            output_len=output_len,
            use_timestamps= True,
            batch_size=64,
            learning_rate=5e-4,
            num_epochs=num_epochs,
            gpus="0",
            callbacks=[sl_callback]
        )

        BasicTSLauncher.launch_training(config)


if __name__ == "__main__":

    # Note: 
    # 1. You need to train the estimation model first.
    # 2. The checkpoint paths of the estimation model are put in the `checkpoint_paths` dictionary.

    train_est_model() 

    est_model_checkpoint_paths = {
        96: "/home/cds/mnt/thongtx/selective-learning/checkpoints/DLinear/ETTh1_100_336_96/ae94700c1226ac6281ae471443fa25ff/DLinear_100.pt",
        192: "/home/cds/mnt/thongtx/selective-learning/checkpoints/DLinear/ETTh1_100_336_192/2389bfc68282f751f2c793984a7979d4/DLinear_100.pt",
        336: "/home/cds/mnt/thongtx/selective-learning/checkpoints/DLinear/ETTh1_100_336_336/e854f4ac31340a1b2c701f6bb4b6403f/DLinear_100.pt",
        720: "/home/cds/mnt/thongtx/selective-learning/checkpoints/DLinear/ETTh1_100_336_720/b1d9a52d99ea134732d90b4ecd082f09/DLinear_100.pt",
    }

    train_main_model(est_model_checkpoint_paths)
