import math
import torch
from torch import nn
import torch.nn.functional as F
from meta_neural_network_architectures import *


def filter_dict(key, params_dict):
    if params_dict is None:
        return None
    res_dict = dict()
    for name, param in params_dict.items():
        bits = name.split('.')
        if key in bits:
            res_dict['.'.join(bits[1:])] = param
    return res_dict


class BatchNorm(nn.Module):
    def __init__(self, num_features, device, args, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, meta_batch_norm=True, no_learnable_params=False,
                 use_per_step_bn_statistics=False):
        super(BatchNorm, self).__init__()
        self.eps = eps
        self.device = device
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.running_mean = nn.Parameter(torch.zeros(num_features), requires_grad=False)
        self.running_var = nn.Parameter(torch.zeros(num_features), requires_grad=False)


    def forward(self, x, num_step, params=None, training=False, backup_running_statistics=False):
        if params:
            weight = params['weight']
            bias = params['bias']
        else:
            weight = self.weight
            bias = self.bias

        if training:
            m1 = x.mean((0, 2, 3))
            m2 = (x ** 2).mean((0, 2, 3))
            self.running_mean.data = (1 - self.momentum) * self.running_mean + self.momentum * m1
            self.running_var.data = (1 - self.momentum) * self.running_var + self.momentum * (m2 - m1 ** 2)
        return F.batch_norm(x, self.running_mean, self.running_var, weight, bias, False,
                            self.momentum, self.eps)


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, num_step, params=None, training=False, backup_running_statistics=False):
        if params:
            weight = params['weight']
            bias = params['bias']
        else:
            weight = self.weight
            bias = self.bias

        return F.conv2d(x, weight, bias=bias, padding=self.padding)


class BottleneckLayer(nn.Module):
    def __init__(self, in_channels, device, args, batch_norm_cls):
        super(BottleneckLayer, self).__init__()
        self.k = 64  # growth rate
        self.bn1 = batch_norm_cls(in_channels, device, args)
        self.conv1 = Conv2d(in_channels, 4 * self.k, (1, 1))
        self.bn2 = batch_norm_cls(4 * self.k, device, args)
        self.conv2 = Conv2d(4 * self.k, self.k, (3, 3), padding=1)

    def forward(self, x, num_step, params=None, training=False, backup_running_statistics=False):
        x = self.bn1(x, num_step, params=filter_dict('bn1', params), training=training,
                     backup_running_statistics=backup_running_statistics)
        x = F.relu(x)
        x = self.conv1(x, num_step, params=filter_dict('conv1', params), training=training)
        x = self.bn2(x, num_step, params=filter_dict('bn2', params), training=training,
                     backup_running_statistics=backup_running_statistics)
        x = F.relu(x)
        x = self.conv2(x, num_step, params=filter_dict('conv2', params), training=training)
        return x


class SqueezeExciteConvLayer(nn.Module):
    def __init__(self, in_channels):
        super(SqueezeExciteConvLayer, self).__init__()
        reduced = max(in_channels // 16, 1)
        self.w1 = nn.Parameter(torch.Tensor(reduced, in_channels))
        self.w2 = nn.Parameter(torch.Tensor(in_channels, reduced))
        nn.init.uniform_(self.w1)
        nn.init.uniform_(self.w2)

    def forward(self, x, num_step, params=None, training=False, backup_running_statistics=False):
        if params:
            w1 = params['w1']
            w2 = params['w2']
        else:
            w1 = self.w1
            w2 = self.w2
        z = x.mean((-2, -1))
        z = F.relu(F.linear(z, w1))
        s = torch.sigmoid(F.linear(z, w2)).unsqueeze(2).unsqueeze(3)
        return s * x


class DenseBlock(nn.Module):
    def __init__(self, in_channels, device, args, batch_norm_cls):
        super(DenseBlock, self).__init__()
        self.se1 = SqueezeExciteConvLayer(in_channels)
        self.bc1 = BottleneckLayer(in_channels, device, args, batch_norm_cls)
        self.se2 = SqueezeExciteConvLayer(self.bc1.k + in_channels)
        self.bc2 = BottleneckLayer(self.bc1.k + in_channels, device, args, batch_norm_cls)
        self.n_out_channels = self.bc2.k + self.bc1.k + in_channels

    def forward(self, x, num_step, params=None, training=False, backup_running_statistics=False):
        y = self.se1(x, num_step, params=filter_dict('se1', params), training=training)
        y = self.bc1(y, num_step, params=filter_dict('bc1', params), training=training,
                     backup_running_statistics=backup_running_statistics)
        x = torch.cat((x, y), 1)
        y = self.se2(x, num_step, params=filter_dict('se2', params), training=training)
        y = self.bc2(y, num_step, params=filter_dict('bc2', params), training=training,
                     backup_running_statistics=backup_running_statistics)
        return torch.cat((x, y), 1)


class DenseBlockUnit(nn.Module):
    def __init__(self, in_channels, device, args, batch_norm_cls):
        super(DenseBlockUnit, self).__init__()
        self.se = SqueezeExciteConvLayer(in_channels)
        self.bc = BottleneckLayer(in_channels, device, args, batch_norm_cls)
        self.n_out_channels = self.bc.k + in_channels

    def forward(self, x, num_step, params=None, training=False, backup_running_statistics=False):
        y = self.se(x, num_step, params=filter_dict('se', params), training=training)
        y = self.bc(y, num_step, params=filter_dict('bc', params), training=training,
                    backup_running_statistics=backup_running_statistics)
        return torch.cat((x, y), 1)


class HighEndEmbedding(nn.Module):
    def __init__(self, device, args, in_channels=3):
        super(HighEndEmbedding, self).__init__()

        self.dbu1 = DenseBlockUnit(3, device, args, BatchNorm)
        self.dbu2 = DenseBlockUnit(self.dbu1.n_out_channels, device, args, BatchNorm)

        n_out2 = max(self.dbu2.n_out_channels // 2, 1)
        self.tr_conv = Conv2d(self.dbu2.n_out_channels, n_out2, (1, 1))
        self.tr_av_pool = nn.AvgPool2d(2, stride=2)

        self.dbu3 = DenseBlockUnit(n_out2, device, args, BatchNorm)
        self.n_out_channels = self.dbu3.n_out_channels

    def forward(self, x, num_step, params=None, training=False, backup_running_statistics=False):
        # First two dense blocks
        x = self.dbu1(x, num_step, params=filter_dict('dbu1', params), training=training,
                      backup_running_statistics=backup_running_statistics)
        x = self.dbu2(x, num_step, params=filter_dict('dbu2', params), training=training,
                      backup_running_statistics=backup_running_statistics)

        # Transition
        x = self.tr_conv(x, num_step, params=filter_dict('tr_conv', params), training=training)
        x = self.tr_av_pool(x)  #

        # 3/4:th dense block (embedding)
        x = self.dbu3(x, num_step, params=filter_dict('dbu3', params), training=training,
                      backup_running_statistics=backup_running_statistics)
        return x


class HighEndClassifier(nn.Module):
    def __init__(self, device, args, in_channels):
        super(HighEndClassifier, self).__init__()
        self.dbu4 = DenseBlockUnit(in_channels, device, args, MetaBatchNormLayer)
        self.weight = nn.Parameter(torch.Tensor(args.num_classes_per_set, self.dbu4.n_out_channels))
        self.bias = nn.Parameter(torch.Tensor(args.num_classes_per_set))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, num_step, params=None, training=False, backup_running_statistics=False):
        x = self.dbu4(x, num_step, params=filter_dict('dbu4', params), training=training,
                      backup_running_statistics=backup_running_statistics).mean((-2, -1))
        if params:
            weight = params['weight']
            bias = params['bias']
        else:
            weight = self.weight
            bias = self.bias
        x = F.linear(x, weight, bias)
        return F.softmax(x, dim=-1)

    def zero_grad(self, params=None):
        if params is None:
            for param in self.parameters():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
        else:
            for name, param in params.items():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
                            params[name].grad = None

