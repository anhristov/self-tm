import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class SelfTM(nn.Module):
    r""" SelfTM: Self-supervised foundation model for template matching (https://www.mdpi.com/2504-2289/9/2/38)

    Args:
        in_chans (int): Number of input image channels. Default: 3
        depths (list): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (list): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
            self,
            in_chans=3,
            depths=[3, 3, 9, 3],
            dims=[96, 192, 384, 768],
            drop_path_rate=0.1,
            layer_scale_init_value=0.0,
    ):
        super().__init__()

        self.dims = dims

        # Downsampling
        self.downsample_layers = (nn.ModuleList())
        stem = nn.Sequential(nn.Conv2d(in_chans, dims[0], kernel_size=3, stride=3), LayerNorm(dims[0], eps=1e-6, data_format="channels_first"), )
        self.downsample_layers.append(stem)
        for i in range(len(dims)-1):
            downsample_layer = nn.Sequential(LayerNorm(dims[i], eps=1e-6, data_format="channels_first"), nn.Conv2d(dims[i], dims[i + 1], kernel_size=3, stride=3), )
            self.downsample_layers.append(downsample_layer)

        # Building blocks
        self.stages = (nn.ModuleList())
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(len(dims)):
            stage = nn.Sequential(*[Block(dim=dims[i], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value, ) for j in range(depths[i])])
            self.stages.append(stage)
            cur += depths[i]

        self.norms = []
        for dim in dims:
            if torch.cuda.is_available():
                self.norms.append(nn.LayerNorm(dim, eps=1e-6).cuda())
            else:
                self.norms.append(nn.LayerNorm(dim, eps=1e-6))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.ConvTranspose2d,)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        features_list = []
        for i in range(len(self.dims)):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

            x_shape = x.shape
            x_norm = x.view(x_shape[0], x_shape[1], x_shape[2] * x_shape[3]).permute(0, 2, 1)
            x_norm = self.norms[i](x_norm)
            x_norm = x_norm.permute(0, 2, 1).view(x_shape)
            features_list.append(x_norm)

        return features_list[::-1]

    def forward(self, x):
        x = self.forward_features(x)
        return x

    def get_feature_maps(self, x):
        _, _, featuremaps = self.forward_features(x)
        return featuremaps

def selftm_small(**kwargs):
    _depths = [3, 9, 3]
    _dims = [128, 256, 512]
    model = SelfTM(depths=_depths, dims=_dims, **kwargs)
    return model, _dims

def selftm_base(**kwargs):
    _depths = [3, 9, 3]
    _dims = [128, 384, 1024]
    model = SelfTM(depths=_depths, dims=_dims, **kwargs)
    return model, _dims

def selftm_large(**kwargs):
    _depths = [3, 9, 3]
    _dims = [128, 512, 2048]
    model = SelfTM(depths=_depths, dims=_dims, **kwargs)
    return model, _dims