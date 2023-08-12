from typing import Callable, Optional, Union, Tuple, List
from functools import partial
from torch.nn import functional as F
import torch
import torch.nn as nn


def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class ConvBNActivation(nn.Sequential):
    """
        类型标注可以使程序的维护性、使用性更高，这一点非常重要；
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]] = 3,
                 stride: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super(ConvBNActivation, self).__init__(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False
            ),
            norm_layer(out_channels),
            activation_layer(inplace=True)
        )


class SqueezeExcitation(nn.Module):
    def __init__(self,
                 input_channels: int,
                 squeeze_factor: int = 4):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = _make_divisible(input_channels // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=squeeze_c, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels=squeeze_c, out_channels=input_channels, kernel_size=1)

    def forward(self, x):
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)
        scale = F.hardsigmoid(scale, inplace=True)
        return scale * x


class InvertedResidualConfig:
    # Stores information listed at Tables 1 and 2 of the MobileNetV3 paper
    def __init__(
        self,
        input_channels: int,
        kernel: int,
        expanded_channels: int,
        out_channels: int,
        use_se: bool,
        activation: str,
        stride: int,
        width_mult: float,
    ):
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride

    @staticmethod
    def adjust_channels(channels: int, width_mult: float):
        return _make_divisible(channels * width_mult, 8)


class InvertedResidual(nn.Module):
    def __init__(self,
                 cnf: InvertedResidualConfig,
                 norm_layer: Callable[..., nn.Module]):
        super(InvertedResidual, self).__init__()

        if cnf.stride not in [1, 2]:
            raise ValueError("illegal stride value")

        self.use_res_connect = (cnf.stride == 1 and cnf.input_channels == cnf.out_channels)

        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        # 在mobilenetV3 large 版本中第一个 bneck 的输入输出channel相等，不需要第一层的1x1卷积
        if cnf.expanded_channels != cnf.input_channels:
            layers.append(ConvBNActivation(cnf.input_channels,
                                           cnf.expanded_channels,
                                           kernel_size=1,
                                           norm_layer=norm_layer,
                                           activation_layer=activation_layer))

        layers.append(ConvBNActivation(cnf.expanded_channels,
                                       cnf.expanded_channels,
                                       kernel_size=cnf.kernel,
                                       stride=cnf.stride,
                                       groups=cnf.expanded_channels,
                                       norm_layer=norm_layer,
                                       activation_layer=activation_layer))

        if cnf.use_se:
            layers.append(SqueezeExcitation(cnf.expanded_channels))

        layers.append(ConvBNActivation(cnf.expanded_channels,
                                       cnf.out_channels,
                                       kernel_size=1,
                                       norm_layer=norm_layer,
                                       activation_layer=nn.Identity))

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels

    def forward(self, x):
        result = self.block(x)
        if self.use_res_connect:
            result += x
        return result


class MobileNetV3(nn.Module):
    """
    MobileNet V3 main class

    Args:
        inverted_residual_setting (List[InvertedResidualConfig]): Network structure
        last_channel (int): The number of channels on the penultimate layer
        num_classes (int): Number of classes
        block (Optional[Callable[..., nn.Module]]): Module specifying inverted residual building block for mobilenet
        norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
        dropout (float): The droupout probability
    """

    def __init__(self,
                 inverted_residual_setting: List[InvertedResidualConfig],
                 last_channel: int,
                 num_classes: int = 1000,
                 block: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 dropout: float = 0.2,
                 ):
        super(MobileNetV3, self).__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, List)
            and all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])
        ):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            # partial模块为高阶函数提供参数设置支持
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers: List[nn.Module] = []

        # first layer
        firstconv_output_c = inverted_residual_setting[0].input_channels
        layers.append(ConvBNActivation(3,
                                       firstconv_output_c,
                                       kernel_size=3,
                                       stride=2,
                                       norm_layer=norm_layer,
                                       activation_layer=nn.Hardswish))
        # inverted residual bloks
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer=norm_layer))

        # last several layers
        lastconv_input_c = inverted_residual_setting[-1].out_channels
        lastconv_output_c = 6 * lastconv_input_c
        layers.append(ConvBNActivation(lastconv_input_c,
                                       lastconv_output_c,
                                       kernel_size=1,
                                       norm_layer=norm_layer,
                                       activation_layer=nn.Hardswish))
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Linear(lastconv_output_c, last_channel),
                                        nn.Hardswish(inplace=True),
                                        nn.Dropout(p=dropout, inplace=True),
                                        nn.Linear(last_channel, num_classes))

        # initial weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


def MobileNetV3_large(num_classes: int = 1000,
                      reduced_tail: bool = False):
    '''
    weight_link:
        url="https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth",

    args:
        num_classes: Model classification number
        reduced_tail: if reduced_tail is True, it reduces the model parameters
    '''

    width_mult = 1.0
    bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_mult=width_mult)

    reduce_divider = 2 if reduced_tail else 1

    Inverted_residual_setting = [
        bneck_conf(16, 3, 16, 16, False, "RE", 1),
        bneck_conf(16, 3, 64, 24, False, "RE", 2),  # C1
        bneck_conf(24, 3, 72, 24, False, "RE", 1),
        bneck_conf(24, 5, 72, 40, True, "RE", 2),  # C2
        bneck_conf(40, 5, 120, 40, True, "RE", 1),
        bneck_conf(40, 5, 120, 40, True, "RE", 1),
        bneck_conf(40, 3, 240, 80, False, "HS", 2),  # C3
        bneck_conf(80, 3, 200, 80, False, "HS", 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1),
        bneck_conf(80, 3, 480, 112, True, "HS", 1),
        bneck_conf(112, 3, 672, 112, True, "HS", 1),
        bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2),  # C4
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1),
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1),
    ]
    last_channel = adjust_channels(1280 // reduce_divider)  # C5

    return MobileNetV3(inverted_residual_setting=Inverted_residual_setting,
                       last_channel=last_channel,
                       num_classes=num_classes)


def MobileNetV3_small(num_classes: int = 1000,
                      reduced_tail: bool = False):
    '''
    weight_link:
        url="https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth",

    args:
        num_classes: Model classification number
        reduced_tail: if reduced_tail is True, it reduces the model parameters
    '''
    width_mult = 1.0
    bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_mult=width_mult)

    reduce_divider = 2 if reduced_tail else 1

    inverted_residual_setting = [
                bneck_conf(16, 3, 16, 16, True, "RE", 2),  # C1
                bneck_conf(16, 3, 72, 24, False, "RE", 2),  # C2
                bneck_conf(24, 3, 88, 24, False, "RE", 1),
                bneck_conf(24, 5, 96, 40, True, "HS", 2),  # C3
                bneck_conf(40, 5, 240, 40, True, "HS", 1),
                bneck_conf(40, 5, 240, 40, True, "HS", 1),
                bneck_conf(40, 5, 120, 48, True, "HS", 1),
                bneck_conf(48, 5, 144, 48, True, "HS", 1),
                bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2),  # C4
                bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1),
                bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1),
    ]
    last_channel = adjust_channels(1024 // reduce_divider)  # C5

    return MobileNetV3(inverted_residual_setting=inverted_residual_setting,
                       last_channel=last_channel,
                       num_classes=num_classes)
