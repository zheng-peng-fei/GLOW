import torch
from torch import nn
from torch.nn import functional as F
from math import log, pi, exp
import numpy as np
from scipy import linalg as la

logabs = lambda x: torch.log(torch.abs(x))


class ActNorm(nn.Module):       #加上均值乘以方差
    def __init__(self, in_channel, logdet=True):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))   #指的是局部归一化（local normalization）中的“loc”参数,表示每个神经元的输出将被减去的均值
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))  #标准差

        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8)) #创建一个名为initialized 的 buffer，并将其初始化为一个 torch.tensor 对象（值为0），数据类型为 torch.uint8 （8 位无符号整数）。这个 buffer 可以被后续的代码访问和修改。
        self.logdet = logdet

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)   #view(input.shape[1], -1):视为2维tensor,第二维的大小自动计算
            mean = (#flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1) 将输入张量(input)展平成二维形式，其中第一维度表示序列长度(sequence_length)，第二维度表示每个序列对应的元素个数，即 (batch_size * height * width)。
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )#mean = flatten.mean(1).unsqueeze(1).unsqueeze(2).unsqueeze(3).permute(1, 0, 2, 3) 计算展平后的二维张量沿着第二个维度(batch_size * height * width)的均值，并对结果进行扩展和置换，得到一个四维张量形式的均值矩阵(mean)。
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )#std = flatten.std(1).unsqueeze(1).unsqueeze(2).unsqueeze(3).permute(1, 0, 2, 3) 计算展平后的二维张量沿着第二个维度(batch_size * height * width)的标准差，并对结果进行扩展和置换，得到一个四维张量形式的标准差矩阵(std)。
            #mean = torch.mean(input, dim=(0, 2, 3), keepdim=True)#用这两个替代更加好用
            #std = torch.std(input, dim=(0, 2, 3), keepdim=True)
            self.loc.data.copy_(-mean)              #减去均值,loc是一个nn.Parameter对象，用data可以获取该参数的值，即一个张量
            self.scale.data.copy_(1 / (std + 1e-6)) #除以方差

    def forward(self, input):
        _, _, height, width = input.shape   #说明输入还是:batch_size、3、height、width

        if self.initialized.item() == 0:   # 最开始为初始化的0，表示第一个batch，要进行数据依赖的初始化；完成后，用1填充，后续就不再执行
            self.initialize(input)          
            self.initialized.fill_(1)       #重置为1，这里指的应该是如果初始化过，那么就无需初始化了

        log_abs = logabs(self.scale)

        logdet = height * width * torch.sum(log_abs)

        if self.logdet:                     # 是否返回对数似然的增量
            return self.scale * (input + self.loc), logdet

        else:
            return self.scale * (input + self.loc)

    def reverse(self, output):
        return output / self.scale - self.loc

# 可逆1X1二维卷积
class InvConv2d(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = torch.randn(in_channel, in_channel)
        q, _ = torch.qr(weight)     #对weight进行qr分解
        weight = q.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(weight)

    def forward(self, input):
        _, _, height, width = input.shape

        out = F.conv2d(input, self.weight)  #weight.shape:torch.Size([3, 3, 1, 1])
        logdet = (
            height * width * torch.slogdet(self.weight.squeeze().double())[1].float()
        )

        return out, logdet

    def reverse(self, output):
        return F.conv2d(
            output, self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3)
        )

# 使用LU分解的快速版可逆二维卷积：可以大大加快计算log|det(W)|的速度，因为分解后只需要计算log|det(W)|=sum(log|s|)即可
class InvConv2dLU(nn.Module):       
    def __init__(self, in_channel):#具体来说，该类构造函数创建了一个随机权重矩阵，并对其进行QR分解和LU分解，然后将得到的矩阵因子存储在模型中。前向传播方法中，调用calc_weight方法计算权重矩阵并使用F.conv2d执行卷积操作。同时，计算出行列式的对数作为输出的附加信息返回。反向传播方法reverse则通过计算反向权重并调用F.conv2d来进行反向传播。
        super().__init__()

        weight = np.random.randn(in_channel, in_channel)
        q, _ = la.qr(weight)        #QR分解
        w_p, w_l, w_u = la.lu(q.astype(np.float32)) #A = PLU，P为置换矩阵，L为下三角、U为上三角；L的对角线元素一定为1
        w_s = np.diag(w_u)                          #获取对角线元素构成数组，一维向量
        w_u = np.triu(w_u, 1)                       #作用：只保留第一条对角线的上三角矩阵，其实就是对角线元素标为0,numpy.triu函数将给定数组（二维）的下三角部分全都变成0，而保留其上三角部分：如果k的值为0，则沿着主对角线进行操作；如果k是正数，则沿着位于主对角线以上的第k条对角线进行操作；如果k是负数，则沿着位于主对角线以下的第k条对角线进行操作。
        u_mask = np.triu(np.ones_like(w_u), 1)      #只保留第一条对角线的上三角矩阵，对应元素全部为1，其他元素变为了0
        l_mask = u_mask.T                           #只保留第-1条对角线的下三角矩阵，对应元素为1，其他元素全部为0
        # 将上述定义的向量转为张量
        w_p = torch.from_numpy(w_p)
        w_l = torch.from_numpy(w_l)
        w_s = torch.from_numpy(w_s)
        w_u = torch.from_numpy(w_u)
        # 将不需要更新的量，将其注册为buffer
        self.register_buffer("w_p", w_p)
        self.register_buffer("u_mask", torch.from_numpy(u_mask))
        self.register_buffer("l_mask", torch.from_numpy(l_mask))
        self.register_buffer("s_sign", torch.sign(w_s))
        self.register_buffer("l_eye", torch.eye(l_mask.shape[0]))
        #需要更新的量
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(logabs(w_s))
        self.w_u = nn.Parameter(w_u)

    def forward(self, input):
        _, _, height, width = input.shape

        weight = self.calc_weight()     #基于分解之后的多个值重构weight

        out = F.conv2d(input, weight)   #卷积计算
        logdet = height * width * torch.sum(self.w_s)   #就为了让这个算的更快，所以才有和上面不一样的分解过程。

        return out, logdet

    def calc_weight(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye) # 保证对角线元素为1
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )

        return weight.unsqueeze(2).unsqueeze(3)

    def reverse(self, output):
        weight = self.calc_weight()

        return F.conv2d(output, weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))

#全零初始化二维卷积
class ZeroConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, padding=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, 3, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input):
        out = F.pad(input, [1, 1, 1, 1], value=1)
        out = self.conv(out)
        out = out * torch.exp(self.scale * 3)

        return out

# 仿射耦合层
class AffineCoupling(nn.Module):    #输入和输出是相同维度的
    def __init__(self, in_channel, filter_size=512, affine=True):
        super().__init__()

        self.affine = affine
        # 就是论文表1中的NN()非线性变换，就是一个模型
        self.net = nn.Sequential(
            nn.Conv2d(in_channel // 2, filter_size, 3, padding=1),  #第一个参数为in_channel,第二个参数为out_channel
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_size, filter_size, 1),
            nn.ReLU(inplace=True),
            ZeroConv2d(filter_size, in_channel if self.affine else in_channel // 2),    #仿射则输出channel=3，否则channel=1
        )
        # net中的第0层即为第一个卷积层
        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()
        # net中的第2层即为第二个卷积层,这里应该是对上面定义的self.net进行初始化
        self.net[2].weight.data.normal_(0, 0.05)
        self.net[2].bias.data.zero_()

    def forward(self, input):
        in_a, in_b = input.chunk(2, 1) # 将input在通道维度上分割为两部分，就是论文表1中的x_a=(n,2,w,h), x_b=(n,1,w,h)      

        if self.affine: # 如果进行仿射,output=cat[in_a,(in_b+t)*s]
            log_s, t = self.net(in_a).chunk(2, 1)   # 经过网络输出log_s和t
            # s = torch.exp(log_s)
            s = F.sigmoid(log_s + 2)    # 论文表1中是取指数，但多个github项目是直接使用sigmoid函数效果更好，训练更稳定
            # out_a = s * in_a + t
            out_b = (in_b + t) * s

            logdet = torch.sum(torch.log(s).view(input.shape[0], -1), 1)

        else:           #如果不进行仿射，output=cat[in_a,in_b+net(in_a)]
            net_out = self.net(in_a)
            out_b = in_b + net_out
            logdet = None

        return torch.cat([in_a, out_b], 1), logdet

    def reverse(self, output):
        out_a, out_b = output.chunk(2, 1)

        if self.affine:
            log_s, t = self.net(out_a).chunk(2, 1)
            # s = torch.exp(log_s)
            s = F.sigmoid(log_s + 2)
            # in_a = (out_a - t) / s
            in_b = out_b / s - t

        else:
            net_out = self.net(out_a)
            in_b = out_b - net_out

        return torch.cat([out_a, in_b], 1)


class Flow(nn.Module):  #把上面定义的Actnorm.Invertible 1 × 1 convolution.和Affine coupling layer.合起来变为一个Flow
    def __init__(self, in_channel, affine=True, conv_lu=True):
        super().__init__()

        self.actnorm = ActNorm(in_channel)

        if conv_lu:
            self.invconv = InvConv2dLU(in_channel)

        else:
            self.invconv = InvConv2d(in_channel)

        self.coupling = AffineCoupling(in_channel, affine=affine)

    def forward(self, input):
        out, logdet = self.actnorm(input)
        out, det1 = self.invconv(out)
        out, det2 = self.coupling(out)

        logdet = logdet + det1
        if det2 is not None:
            logdet = logdet + det2

        return out, logdet

    def reverse(self, output):
        input = self.coupling.reverse(output)
        input = self.invconv.reverse(input)
        input = self.actnorm.reverse(input)

        return input

# 这段代码实现的是计算给定高斯分布（均值为mean，标准差的对数为log_sd）下某个数据点x的概率密度函数值（对数形式
def gaussian_log_p(x, mean, log_sd):    #根据数据和均值、方差，计算x的概率
    return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)


def gaussian_sample(eps, mean, log_sd): #这段代码实现的是从高斯分布中采样的操作。其中，eps表示来自标准正态分布的噪声向量，mean表示高斯分布的均值，log_sd表示标准差的对数
    return mean + torch.exp(log_sd) * eps


class Block(nn.Module):
    '''每个block包含多个flow'''
    def __init__(self, in_channel, n_flow, split=True, affine=True, conv_lu=True):
        super().__init__()

        squeeze_dim = in_channel * 4

        self.flows = nn.ModuleList()    #nn.ModuleList()函数在PyTorch中用于将多个神经网络模块组合成一个列表。
        for i in range(n_flow):
            self.flows.append(Flow(squeeze_dim, affine=affine, conv_lu=conv_lu))

        self.split = split

        if split:           #前几个Block是split=True,后几个Block是split=False
            #print(in_channel)   #3然后是6然后是12
            self.prior = ZeroConv2d(in_channel * 2, in_channel * 4)

        else:
            #print(in_channel)   #24
            self.prior = ZeroConv2d(in_channel * 4, in_channel * 8)

    def forward(self, input):
        b_size, n_channel, height, width = input.shape
        # 对输入的通道数进行扩增，对空间进行缩小，即通道数扩增至四倍，长和宽都变为原来一半；就是论文中图2b中的squeeze操作
        squeezed = input.view(b_size, n_channel, height // 2, 2, width // 2, 2)
        squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
        out = squeezed.contiguous().view(b_size, n_channel * 4, height // 2, width // 2)#channel数目变为原来的4倍
        #print(out.shape)        #torch.Size([4, 12, 128, 128]) torch.Size([4, 24, 64, 64]) torch.Size([4, 48, 32, 32]) torch.Size([4, 96, 16, 16])
        logdet = 0

        for flow in self.flows: #经过多个flow
            out, det = flow(out)
            logdet = logdet + det
        #print(out.shape)   同上，保持不变  #torch.Size([4, 12, 128, 128]) torch.Size([4, 24, 64, 64]) torch.Size([4, 48, 32, 32]) torch.Size([4, 96, 16, 16])
        if self.split:                                  #split表示还没到最后一个模块
            out, z_new = out.chunk(2, 1)    #如out是torch.Size([4, 12, 128, 128])，那么out和z_new的形状都为torch.Size([4, 6, 128, 128])  
            mean, log_sd = self.prior(out).chunk(2, 1)
            #print(mean.shape)   #torch.Size([4, 6, 128, 128]) torch.Size([4, 12, 64, 64]) torch.Size([4, 24, 32, 32])
            #print(log_sd.shape) #同上
            log_p = gaussian_log_p(z_new, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)

        else:
            zero = torch.zeros_like(out)
            mean, log_sd = self.prior(zero).chunk(2, 1)
            #print(mean.shape)  #torch.Size([4, 96, 16, 16])
            #print(log_sd.shape) #torch.Size([4, 96, 16, 16])
            log_p = gaussian_log_p(out, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)
            z_new = out
        #print(out.shape)        #torch.Size([4, 6, 128, 128]) torch.Size([4, 12, 64, 64]) torch.Size([4, 24, 32, 32]) torch.Size([4, 96, 16, 16])
        return out, logdet, log_p, z_new

    def reverse(self, output, eps=None, reconstruct=False):
        input = output

        if reconstruct:
            if self.split:
                input = torch.cat([output, eps], 1)

            else:
                input = eps

        else:
            if self.split:
                mean, log_sd = self.prior(input).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                input = torch.cat([output, z], 1)

            else:
                zero = torch.zeros_like(input)
                # zero = F.pad(zero, [1, 1, 1, 1], value=1)
                mean, log_sd = self.prior(zero).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                input = z

        for flow in self.flows[::-1]:
            input = flow.reverse(input)

        b_size, n_channel, height, width = input.shape

        unsqueezed = input.view(b_size, n_channel // 4, 2, 2, height, width)
        unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3)
        unsqueezed = unsqueezed.contiguous().view(
            b_size, n_channel // 4, height * 2, width * 2
        )

        return unsqueezed


class Glow(nn.Module):
    def __init__(
        self, in_channel, n_flow, n_block, affine=True, conv_lu=True
    ):
        super().__init__()

        self.blocks = nn.ModuleList()
        n_channel = in_channel
        for i in range(n_block - 1):
            self.blocks.append(Block(n_channel, n_flow, affine=affine, conv_lu=conv_lu))
            n_channel *= 2
        self.blocks.append(Block(n_channel, n_flow, split=False, affine=affine))

    def forward(self, input):
        log_p_sum = 0
        logdet = 0
        out = input
        z_outs = []

        for block in self.blocks:
            out, det, log_p, z_new = block(out)
            z_outs.append(z_new)
            logdet = logdet + det

            if log_p is not None:
                log_p_sum = log_p_sum + log_p

        return log_p_sum, logdet, z_outs

    def reverse(self, z_list, reconstruct=False):
        for i, block in enumerate(self.blocks[::-1]):
            if i == 0:
                input = block.reverse(z_list[-1], z_list[-1], reconstruct=reconstruct)

            else:
                input = block.reverse(input, z_list[-(i + 1)], reconstruct=reconstruct)

        return input
