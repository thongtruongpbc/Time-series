import torch
import torch.nn as nn


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str, mask=None):
        if mode == "norm":
            self._get_statistics(x, mask=mask)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x, mask=None):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if mask == None:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
            self.stdev = torch.sqrt(
                torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
            ).detach()

        else:
            sum_x = torch.sum(x * mask, dim=dim2reduce, keepdim=True)
            count_x = torch.sum(mask, dim=dim2reduce, keepdim=True)
            self.mean = (sum_x / (count_x + self.eps)).detach()
            var_x = torch.sum(
                ((x - self.mean) * mask) ** 2, dim=dim2reduce, keepdim=True
            )
            self.stdev = torch.sqrt(var_x / (count_x + self.eps) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight.to(x.device)
            x = x + self.affine_bias.to(x.device)
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias.to(x.device)
            x = x / (self.affine_weight.to(x.device) + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x
