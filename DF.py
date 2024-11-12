import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from typing import Optional
from torch.optim.lr_scheduler import StepLR
#结构是图片进DF进ResNet50
class DilateAttention(nn.Module):#分散注意力
    """
    实现扩张注意力机制的模块。
    """
    def __init__(self, head_dim, qk_scale=None, attn_drop=0, kernel_size=3, dilation=1):
        super().__init__()  # 初始化基类
        # 设置注意力头的维度，如果没有提供qk_scale，则使用默认的缩放系数
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.kernel_size = kernel_size  # 卷积核大小
        self.dilation = dilation  # 扩张率
        self.unfold = nn.Unfold(kernel_size, dilation, dilation * (kernel_size - 1) // 2, 1)  # 使用unfold进行扩张操作
        self.attn_drop = nn.Dropout(attn_drop)  # 注意力系数的dropout

    def forward(self, q, k, v):
        # 输入的维度是(B, C//3, H, W)，分别是批量大小、通道数、高度和宽度
        B, d, H, W = q.shape
        # 重塑和调整查询、键、值的维度以适应操作
        q = q.reshape([B, d // self.head_dim, self.head_dim, 1, H * W]).permute(0, 1, 4, 3, 2)
        k = self.unfold(k).reshape(
            [B, d // self.head_dim, self.head_dim, self.kernel_size * self.kernel_size, H * W]).permute(0, 1, 4, 2, 3)
        # 计算注意力权重，并进行缩放
        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)  # 应用softmax进行归一化
        attn = self.attn_drop(attn)  # 应用dropout
        v = self.unfold(v).reshape(
            [B, d // self.head_dim, self.head_dim, self.kernel_size * self.kernel_size, H * W]).permute(0, 1, 4, 3, 2)
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, d)  # 计算输出
        return x  # 返回处理后的张量

class MultiDilatelocalAttention(nn.Module):
    """
    多尺度扩张局部注意力机制的实现。
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., kernel_size=3, dilation=[2, 3]):
        super().__init__()  # 初始化基类
        self.dim = dim
        self.num_heads = num_heads  # 注意力头数
        head_dim = dim // num_heads  # 每个头的维度
        self.dilation = dilation  # 扩张率列表
        self.kernel_size = kernel_size  # 卷积核大小
        self.scale = qk_scale or head_dim ** -0.5  # 缩放系数
        self.num_dilation = len(dilation)  # 扩张级数
        assert num_heads % self.num_dilation == 0, "注意力头数必须能被扩张级数整除"
        # 初始化多个DilateAttention模块
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)  # 生成QKV的卷积
        self.dilate_attention = nn.ModuleList(
            [DilateAttention(head_dim, qk_scale, attn_drop, kernel_size, dilation[i])
             for i in range(self.num_dilation)])
        self.proj = nn.Linear(dim, dim)  # 输出投影层
        self.proj_drop = nn.Dropout(proj_drop)  # 投影层的dropout

    def forward(self, x):
        B, H, W, C = x.shape  # 输入的维度
        x = x.permute(0, 3, 1, 2)  # 调整维度以适应卷积层
        qkv = self.qkv(x).reshape(B, 3, self.num_dilation, C // self.num_dilation, H, W).permute(2, 1, 0, 3, 4, 5)
        x = x.reshape(B, self.num_dilation, C // self.num_dilation, H, W).permute(1, 0, 3, 4, 2)
        # 应用所有的扩张注意力模块
        out = []  # 使用列表收集每个 dilate_attention 的输出
        for i in range(self.num_dilation):
            out.append(self.dilate_attention[i](qkv[i][0], qkv[i][1], qkv[i][2]))  # 收集结果而不是修改 x[i]
        x = torch.stack(out, dim=0).permute(1, 2, 3, 0, 4).reshape(B, H, W, C)  # 将输出堆叠并调整形状
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

if __name__ == '__main__':
    x=torch.ones([1,3,224,224])
    df=MultiDilatelocalAttention(dim=224)
    y=df(x)
    print(y.shape)