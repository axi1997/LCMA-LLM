import re
import math
import torch
from torch import nn
from functools import partial
from timm.layers.norm_act import LayerNormAct2d
from torchvision.ops.misc import SqueezeExcitation as SElayer
from torchvision.models.mobilenetv3 import InvertedResidual, InvertedResidualConfig

class FeatureProjectionLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Mish(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class PoolDownLayer(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.dwn = nn.AdaptiveAvgPool2d(shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, channels = x.shape
        height = int(math.sqrt(num_tokens))
        assert height * height == num_tokens
        x = x.permute(0, 2, 1).reshape(batch_size, channels, height, height)
        x = self.dwn(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class PositionalEncodingConv(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(
            input_dim, output_dim,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=True,
            groups=output_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, channels = x.shape
        height = int(math.sqrt(num_tokens))
        assert height * height == num_tokens
        x_reshaped = x.transpose(1, 2).view(batch_size, channels, height, height)
        x = self.peg(x_reshaped) + x_reshaped
        x = x.flatten(2).transpose(1, 2)
        return x


class DCAPNetProjector(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        in_dim, out_dim = config.mm_hidden_size, config.hidden_size
        self.mlp = FeatureProjectionLayer(in_dim, out_dim)
        self.dwn = PoolDownLayer((14, 14))
        self.peg = PositionalEncodingConv(out_dim, out_dim, stride=1)

    def forward(self, x):
        x = self.mlp(x)
        x = self.dwn(x)
        x = self.peg(x)
        return x


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)
    elif projector_type.startswith('mlp2x_gelu'):
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.hidden_size, config.hidden_size))
            return nn.Sequential(*modules)
    elif projector_type.startswith('dcapnet'):
        return DCAPNetProjector(config)
    raise ValueError(f'Unknown projector type: {projector_type}')
