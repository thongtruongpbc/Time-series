import os
import torch
from models import (
    Autoformer,
    Transformer,
    TimesNet,
    Nonstationary_Transformer,
    DLinear,
    FEDformer,
    Informer,
    LightTS,
    Reformer,
    ETSformer,
    Pyraformer,
    PatchTST,
    MICN,
    Crossformer,
    FiLM,
    iTransformer,
    Koopa,
    TiDE,
    FreTS,
    TimeMixer,
    TSMixer,
    SegRNN,
    MambaSimple,
    TemporalFusionTransformer,
    SCINet,
    PAttn,
    TimeXer,
    WPMixer,
    MultiPatchFormer,
    KANAD,
    MSGNet,
    TimeFilter,
    Sundial,
    TimeMoE,
    Chronos,
    Moirai,
    TiRex,
    TimesFM,
    Chronos2,
    iTransformer_retrieval,
    Autoformer_retrieval,
    TimesNet_retrieval,
    Transformer_retrieval,
    Crossformer_retrieval,
)


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            "TimesNet": TimesNet,
            "TimesNet_retrieval": TimesNet_retrieval,
            "Autoformer": Autoformer,
            "Autoformer_retrieval": Autoformer_retrieval,
            "Transformer": Transformer,
            "Transformer_retrieval": Transformer_retrieval,
            "Nonstationary_Transformer": Nonstationary_Transformer,
            "DLinear": DLinear,
            "FEDformer": FEDformer,
            "Informer": Informer,
            "LightTS": LightTS,
            "Reformer": Reformer,
            "ETSformer": ETSformer,
            "PatchTST": PatchTST,
            "Pyraformer": Pyraformer,
            "MICN": MICN,
            "Crossformer": Crossformer,
            "FiLM": FiLM,
            "iTransformer": iTransformer,
            "iTransformer_retrieval": iTransformer_retrieval,
            "Koopa": Koopa,
            "TiDE": TiDE,
            "FreTS": FreTS,
            "MambaSimple": MambaSimple,
            "TimeMixer": TimeMixer,
            "TSMixer": TSMixer,
            "SegRNN": SegRNN,
            "TemporalFusionTransformer": TemporalFusionTransformer,
            "SCINet": SCINet,
            "PAttn": PAttn,
            "TimeXer": TimeXer,
            "WPMixer": WPMixer,
            "MultiPatchFormer": MultiPatchFormer,
            "KANAD": KANAD,
            "MSGNet": MSGNet,
            "TimeFilter": TimeFilter,
            "Sundial": Sundial,
            "TimeMoE": TimeMoE,
            "Chronos": Chronos,
            "Moirai": Moirai,
            "TiRex": TiRex,
            "TimesFM": TimesFM,
            "Chronos2": Chronos2,
            "Crossformer_retrieval": Crossformer_retrieval,
        }
        if args.model == "Mamba":
            print("Please make sure you have successfully installed mamba_ssm")
            from models import Mamba

            self.model_dict["Mamba"] = Mamba

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _build_model_emb(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu and self.args.gpu_type == "cuda":
            os.environ["CUDA_VISIBLE_DEVICES"] = (
                str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            )
            device = torch.device("cuda:{}".format(self.args.gpu))
            print("Use GPU: cuda:{}".format(self.args.gpu))
        elif self.args.use_gpu and self.args.gpu_type == "mps":
            device = torch.device("mps")
            print("Use GPU: mps")
        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
