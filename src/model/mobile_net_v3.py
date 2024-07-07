import torch
from torch import nn
import torch.nn.functional as F

class MobileNetV2(nn.Module):

    def __init__(
        self, input_channel, increased_channel, result_channel,
        kernel_size, activation_layer,
        is_se, stride= 1,
        padding= 1):

        super().__init__()
        self.use_skip_conn = stride == 1 and input_channel == result_channel
        self.is_se = is_se

        self.neural_network(input_channel, increased_channel, result_channel, kernel_size, activation_layer, is_se, stride, padding)


    
    def neural_network(self, input_channel, increased_channel, result_channel, kernel_size, activation_layer, is_se, stride, padding):
        self.standard_conv = nn.Sequential(
            nn.Conv2d(input_channel, increased_channel, kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(increased_channel, track_running_stats=False),
            activation_layer(),
        )

        self.cnn_layer = nn.Sequential(
            nn.Conv2d(increased_channel, increased_channel,
                      kernel_size=kernel_size, stride=stride,
                      groups=increased_channel, padding=padding, bias=False),
            nn.BatchNorm2d(increased_channel, track_running_stats=False),
        )

        self.squeeze_excitation = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(increased_channel, input_channel, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(input_channel, increased_channel, kernel_size=1),
            nn.Hardswish(),
        ) if is_se else None

        self.pointwise_conv = nn.Sequential( 
            nn.Conv2d(increased_channel, result_channel, kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(result_channel, track_running_stats=False)
        )

    def forward(self, input_value):

        output = self.standard_conv(input_value)
        depth_output = self.cnn_layer(output)

        if self.is_se:
            output = self.squeeze_excitation(depth_output)
            output = output * depth_output
        else:
            output = depth_output

        output = self.pointwise_conv(output)
        if self.use_skip_conn:
            output = output + input_value

        return output