import torch
import torch.nn as nn
import os
from models.Autoformer import Model as Autoformer_Model
from models.TimesNet import Model as TimesNet_Model
from models.Crossformer import Model as Crossformer_Model
from models.DLinear import Model as DLinear_Model


class Freeze_Backbone(nn.Module):
    def __init__(self, args):
        super(Freeze_Backbone, self).__init__()
        self.args = args
        self.device = args.device

        self.model_dict = {
            "TimesNet": TimesNet_Model,
            "Autoformer": Autoformer_Model,
            "Crossformer": Crossformer_Model,
            "DLinear": DLinear_Model,
        }

        self.freeze_model = self._build_freeze_model()
        # self._initialize_lazy_layers()

        checkpoint_path = os.path.join(
            "imputation/checkpoints_imputation",
            args.setting,
            "checkpoint.pth",
        )

        if os.path.exists(checkpoint_path):
            state = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )
            self.freeze_model.load_state_dict(state, strict=False)
            print(f"-> Successfully loaded frozen model: {checkpoint_path}")
        else:
            print(f"-> Warning: Checkpoint not found at {checkpoint_path}")

        self.freeze_model.to(self.device)
        self.freeze_model.eval()

        for param in self.freeze_model.parameters():
            param.requires_grad = False

    def _initialize_lazy_layers(self):
        """
        Forces initialization of lazy modules by passing a dummy batch through the model.
        """
        # Create dummy inputs based on your dataset dimensions
        # Assuming typical shapes [batch, seq_len, features]
        dummy_x = torch.randn(1, self.args.seq_len, self.args.enc_in).to(self.device)
        dummy_mark = torch.randn(1, self.args.seq_len, self.args.c_out).to(self.device)

        self.freeze_model.to(self.device)
        with torch.no_grad():
            try:
                # one forward pass to set the shapes
                self.freeze_model(dummy_x, dummy_mark, dummy_x, dummy_mark)
            except Exception as e:
                print(f"-> Note: Lazy initialization pass finished with notice: {e}")

    def _build_freeze_model(self):
        model_name = self.args.model_emb
        if model_name not in self.model_dict:
            raise ValueError(f"Model {model_name} is not supported")
        model = self.model_dict[model_name](self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.freeze_model(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
