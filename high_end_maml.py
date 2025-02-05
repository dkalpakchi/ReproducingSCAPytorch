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
    return res_dict or None


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
        self.use_per_step_bn_statistics = use_per_step_bn_statistics
        self.running_mean = nn.Parameter(torch.zeros(num_features), requires_grad=False)
        self.running_var = nn.Parameter(torch.ones(num_features), requires_grad=False)
        self.backup_running_mean = torch.zeros(self.running_mean.shape)
        self.backup_running_var = torch.ones(self.running_var.shape)


    def forward(self, x, num_step, params=None, training=False, backup_running_statistics=False):
        if params:
            weight = params['weight']
            bias = params['bias']
        else:
            weight = self.weight
            bias = self.bias

        if backup_running_statistics and self.use_per_step_bn_statistics:
            self.backup_running_mean.data = copy(self.running_mean.data)
            self.backup_running_var.data = copy(self.running_var.data)

        if training:
            m1 = x.mean((0, 2, 3))
            m2 = (x ** 2).mean((0, 2, 3))
            self.running_mean.data = (1 - self.momentum) * self.running_mean + self.momentum * m1
            self.running_var.data = (1 - self.momentum) * self.running_var + self.momentum * (m2 - m1 ** 2)
        return F.batch_norm(x, self.running_mean, self.running_var, weight, bias, False,
                            self.momentum, self.eps)

    def restore_backup_stats(self):
        """
        Resets batch statistics to their backup values which are collected after each forward pass.
        """
        if self.use_per_step_bn_statistics:
            self.running_mean = nn.Parameter(self.backup_running_mean.to(device=self.device), requires_grad=False)
            self.running_var = nn.Parameter(self.backup_running_var.to(device=self.device), requires_grad=False)


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
        # Same as MAML++
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

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
        self.norm_layer_1 = batch_norm_cls(in_channels, device, args)
        self.conv1 = Conv2d(in_channels, 4 * self.k, (1, 1))
        self.norm_layer_2 = batch_norm_cls(4 * self.k, device, args)
        self.conv2 = Conv2d(4 * self.k, self.k, (3, 3), padding=1)

    def forward(self, x, num_step, params=None, training=False, backup_running_statistics=False):
        x = self.norm_layer_1(x, num_step, params=filter_dict('norm_layer_1', params), training=training,
                     backup_running_statistics=backup_running_statistics)
        x = F.relu(x)
        x = self.conv1(x, num_step, params=filter_dict('conv1', params), training=training)
        x = self.norm_layer_2(x, num_step, params=filter_dict('norm_layer_2', params), training=training,
                     backup_running_statistics=backup_running_statistics)
        x = F.relu(x)
        x = self.conv2(x, num_step, params=filter_dict('conv2', params), training=training)
        return x

    def restore_backup_stats(self):
        """
        Resets batch statistics to their backup values which are collected after each forward pass.
        """
        self.norm_layer_1.restore_backup_stats()
        self.norm_layer_2.restore_backup_stats()


class SqueezeExciteConvLayer(nn.Module):
    def __init__(self, in_channels):
        super(SqueezeExciteConvLayer, self).__init__()
        reduced = max(in_channels // 16, 1)
        self.w1 = nn.Parameter(torch.Tensor(reduced, in_channels))
        self.w2 = nn.Parameter(torch.Tensor(in_channels, reduced))
        #nn.init.kaiming_normal_(self.w1)
        #nn.init.kaiming_normal_(self.w2)
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)

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

    def restore_backup_stats(self):
        """
        Resets batch statistics to their backup values which are collected after each forward pass.
        """
        self.bc.restore_backup_stats()


class HighEndEmbedding(nn.Module):
    def __init__(self, device, args, in_channels=3):
        super(HighEndEmbedding, self).__init__()

        self.dbu1 = DenseBlockUnit(3, device, args, BatchNorm)
        self.dbu2 = DenseBlockUnit(self.dbu1.n_out_channels, device, args, BatchNorm)

        n_out2 = max(self.dbu2.n_out_channels // 2, 1)
        self.tr_conv = Conv2d(self.dbu2.n_out_channels, n_out2, (1, 1))
        self.tr_av_pool = nn.AvgPool2d(4, stride=4)

        self.dbu3 = DenseBlockUnit(n_out2, device, args, BatchNorm)
        self.n_out_channels = self.dbu3.n_out_channels

        self.args = args

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

        # Receptive field change
        return F.adaptive_avg_pool2d(x, (5, 5))


class HighEndClassifier(nn.Module):
    def __init__(self, device, args, in_channels):
        super(HighEndClassifier, self).__init__()
        self.dbu4 = DenseBlockUnit(in_channels, device, args, MetaBatchNormLayer)
        self.weight = nn.Parameter(torch.Tensor(args.num_classes_per_set, self.dbu4.n_out_channels))
        self.bias = nn.Parameter(torch.Tensor(args.num_classes_per_set))
        self.reset_parameters()

    def reset_parameters(self):
        ## Same as MAML++ linear does not work!!!
        #nn.init.xavier_uniform_(self.weight)
        #nn.init.zeros_(self.bias)
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
        y = F.linear(x, weight, bias)  # Seems they don't use softmax in the end???
        return y

    def restore_backup_stats(self):
        """
        Resets batch statistics to their backup values which are collected after each forward pass.
        """
        self.dbu4.restore_backup_stats()

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

