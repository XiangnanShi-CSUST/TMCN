import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange
from Mamba import VSSBlock as MBaseFeatureExtraction
from FMamba import CrossMambaFusionBlock, ConcatMambaFusionBlock

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + \
        torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class AttentionBase(nn.Module):
    def __init__(self,
                 dim,   
                 num_heads=8,
                 qkv_bias=False,):
        super(AttentionBase, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv1 = nn.Conv2d(dim, dim*3, kernel_size=1, bias=qkv_bias)
        self.qkv2 = nn.Conv2d(dim*3, dim*3, kernel_size=3, padding=1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        b, c, h, w = x.shape
        qkv = self.qkv2(self.qkv1(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.proj(out)
        return out
    
class Mlp(nn.Module):
    # 这段代码定义了一个多层感知机（MLP）模块，用于在视觉Transformer、MLP-Mixer和相关网络中进行特征变换
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, 
                 in_features, 
                 hidden_features=None, 
                 ffn_expansion_factor = 2,
                 # hidden_features：隐藏层特征的维度大小，默认为None，即与输入特征维度相同。
                 # ffn_expansion_factor：前馈神经网络（FeedForward）的扩展因子，默认为2。
                 bias = False):
        super().__init__()
        hidden_features = int(in_features*ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            in_features, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, in_features, kernel_size=1, bias=bias)
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class BaseFeatureExtraction(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 ffn_expansion_factor=1.,
                 qkv_bias=False, ):
        super(BaseFeatureExtraction, self).__init__()
        self.norm1 = LayerNorm(dim, 'WithBias')
        self.attn = AttentionBase(dim, num_heads=num_heads, qkv_bias=qkv_bias, )
        self.norm2 = LayerNorm(dim, 'WithBias')
        self.mlp = Mlp(in_features=dim,
                       ffn_expansion_factor=ffn_expansion_factor, )


    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x




class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.bottleneckBlock = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # dw
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            # nn.BatchNorm2d(oup),
        )
    def forward(self, x):
        return self.bottleneckBlock(x)

class DetailNode(nn.Module):
    def __init__(self):
        super(DetailNode, self).__init__()
        # Scale is Ax + b, i.e. affine transformation
        self.theta_phi = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_rho = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_eta = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.shffleconv = nn.Conv2d(64, 64, kernel_size=1,
                                    stride=1, padding=0, bias=True)
    def separateFeature(self, x):
        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        return z1, z2
    def forward(self, z1, z2):
        z1, z2 = self.separateFeature(
            self.shffleconv(torch.cat((z1, z2), dim=1)))
        z2 = z2 + self.theta_phi(z1)
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)
        return z1, z2

class MDetailFeatureExtraction(nn.Module):
    def __init__(self, model_clip, num_layers=3, hidden_dim: int = 64, norm_layer=nn.LayerNorm):
        super(MDetailFeatureExtraction, self).__init__()
        INNmodules = [DetailNode() for _ in range(num_layers)]
        self.net = nn.Sequential(*INNmodules)
        self.scale2 = nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = DetailChannelAttentionBlock(hidden_dim)
        self.norm2 = norm_layer(hidden_dim)

        self.model_clip = model_clip
        self.prompt_guidance_4 = FeatureWiseAffine(in_channels=512, out_channels=64)

    def forward(self, x, text):
        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        for layer in self.net:
            z1, z2 = layer(z1, z2)
        x = torch.cat((z1, z2), dim=1)
        x = x.permute(0, 2, 3, 1)
        y = self.conv_blk(self.norm2(x).permute(0, 3, 1, 2).contiguous()) + (x * self.scale2).permute(0, 3, 1,
                                                                                                      2).contiguous()

        b = y.shape[0]
        text_features = self.get_text_feature(text.expand(b, -1)).to(y.dtype)
        y = self.prompt_guidance_4(y, text_features)
        return y

    @torch.no_grad()
    def get_text_feature(self, text):
        text_feature = self.model_clip.encode_text(text)
        return text_feature

class DetailFeatureExtraction(nn.Module):
    def __init__(self, num_layers=3, hidden_dim: int = 64, norm_layer=nn.LayerNorm):
        super(DetailFeatureExtraction, self).__init__()
        INNmodules = [DetailNode() for _ in range(num_layers)]
        self.net = nn.Sequential(*INNmodules)
        self.scale2 = nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = DetailChannelAttentionBlock(hidden_dim)
        self.norm2 = norm_layer(hidden_dim)

    def forward(self, x):

        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]

        for layer in self.net:
            z1, z2 = layer(z1, z2)
        x = torch.cat((z1, z2), dim=1)+x

        return x

# =============================================================================

# =============================================================================
import numbers
##########################################################################
## Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')
    # 这行代码使用了 `rearrange` 函数，它的作用是重新排列张量的维度顺序。具体来说，这行代码将输入张量 `x` 从形状 `'b c h w'`（batch, channels, height, width）重新排列成了 `'b (h w) c'`（batch, height*width, channels）的形状。
    #
    # - `'b c h w'` 表示原始张量的形状，其中：
    #   - `b` 是 batch size（批大小），
    #   - `c` 是 channels（通道数），
    #   - `h` 是 height（高度），
    #   - `w` 是 width（宽度）。
    # - `'b (h w) c'` 表示重新排列后的形状，其中：
    #   - `b` 仍然是 batch size，
    #   - `(h w)` 表示将原始的 height 和 width 拼接在一起，形成一个新的维度，
    #   - `c` 仍然是 channels。
    #
    # 这种重新排列的操作通常在深度学习中用于将二维的特征图展平成一维向量，以便输入全连接层等需要一维输入的层。


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
    # 这个 `to_4d` 函数将一个形状为 `(batch_size, h*w, channels)` 的 3D 张量重排列成形状为 `(batch_size, channels, h, w)` 的 4D 张量，其中 `h` 和 `w` 是输出张量的高度和宽度。
    #
    # 它使用了 `rearrange` 函数来实现张量的重排列。在这个函数中，`'b (h w) c -> b c h w'` 是重排列的模式字符串，它指定了如何将原始张量重新排列为目标形状。 `h` 和 `w` 参数用于指定输出张量的高度和宽度。


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        # 总体而言，BiasFree_LayerNorm 实现了一个无偏置的 Layer Normalization 操作，
        # 其核心在于使用方差的平方根作为缩放因子，通过可学习的参数 self.weight 控制归一化的过程。
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        # 初始化方法接收一个参数 normalized_shape，它表示归一化的形状。
        # 根据 normalized_shape 的类型，将其转换为 torch.Size 类型的数据。

        assert len(normalized_shape) == 1
        # 确保 normalized_shape 的长度为 1，即保证是一维的归一化形状。

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape
        # 创建了一个可学习的参数 self.weight，这个参数被初始化为一个与 normalized_shape 大小相匹配的张量，全部元素的值为 1。
        # 这个参数在 Layer Normalization 过程中用于缩放输入张量。

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight
        #  X为输入张量
        # 					Var(x) 是输入张量在最后一个维度上的方差，
        # 					ϵ 是一个非常小的数，用于数值稳定性，
        # 					weight 是可学习的缩放参数。
        # 					这个公式对输入张量 x 进行归一化，然后乘以可学习参数 self.weight 进行缩放。
        #
        # forward 方法定义了模型的前向传播过程。
        # 接收输入张量 x，它是模型要进行归一化处理的数据。
        # 计算输入张量 x 在最后一个维度上的方差 sigma，设置 unbiased=False 表示使用有偏的方式计算方差。
        # 对输入张量 x 进行归一化处理，使用方差的平方根作为缩放因子，同时乘以 self.weight 参数对输入进行缩放。
        # 最终返回归一化后的张量。


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        # 初始化方法接收一个参数 normalized_shape，表示归一化的形状。
        # 将 normalized_shape 转换为 torch.Size 类型的数据，确保其长度为 1，即保证是一维的归一化形状。
        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

        # 创建了两个可学习的参数：self.weight 和 self.bias。
        # self.weight 是一个张量参数，全部元素的值为 1，用于在归一化过程中缩放输入张量。
        # self.bias 是一个张量参数，全部元素的值为 0，用于在归一化过程中添加偏置。

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias
        # μ 是输入张量在最后一个维度上的均值，
        # 					bias 是可学习的偏置参数
        # 					这个公式对输入张量 x 进行归一化，并在归一化后的结果上添加了可学习的偏置参数bias
        #
        # forward 方法定义了模型的前向传播过程。
        # 接收输入张量 x，即要进行归一化处理的数据。
        # 计算输入张量 x 在最后一个维度上的均值 mu 和方差 sigma。
        # 对输入张量 x 进行归一化处理，通过减去均值 mu 并除以方差的平方根，同时乘以 self.weight 参数进行缩放，并添加上 self.bias 参数作为偏置。
        # 最终返回归一化和偏置处理后的张量。
class LayerNorm(nn.Module):
    # 这段代码是一个自定义的 Layer Normalization 类 `LayerNorm`，其根据输入的 `LayerNorm_type` 参数选择性地初始化一个特定类型的 Layer Normalization 模型，可以是无偏置的 `BiasFree_LayerNorm` 或有偏置的 `WithBias_LayerNorm`。
    #
    # 在前向传播函数中，首先获取输入张量 `x` 的高度和宽度，然后通过 `to_3d` 函数将输入张量转换为三维张量，再将其传递给选定的 Layer Normalization 模型 `self.body` 进行处理。处理完成后，使用 `to_4d` 函数将处理后的三维张量重新转换为四维张量，并返回结果。
    #
    # 综合上述解读，`return to_4d(self.body(to_3d(x)), h, w)` 的作用是将输入张量 `x` 经过选择的 Layer Normalization 处理后，将其形状重新调整为原始形状，并返回处理结果。
    def __init__(self, dim, LayerNorm_type):
        # 根据 LayerNorm_type 参数的值，选择性地初始化一个具体的 Layer Normalization 模型（BiasFree_LayerNorm 或 WithBias_LayerNorm）。
        # 选择在前向传播时应用的具体的 Layer Normalization 方法。
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        # 这行代码 `h, w = x.shape[-2:]` 是用来获取张量 `x` 在倒数第二和倒数第一维度上的尺寸，通常用于处理图像数据。在图像数据中，通常有两个维度表示高度（height）和宽度（width），因此这行代码可以方便地获取图像的高度和宽度。
        #
        # - `x.shape` 返回张量 `x` 的形状，是一个包含各个维度大小的元组。
        # - `[-2:]` 表示从倒数第二个维度开始到最后一个维度（即最后一个维度和倒数第二个维度）的切片。
        # - `h, w =` 则是将切片得到的两个尺寸分别赋值给变量 `h` 和 `w`。
        #
        # 这样，变量 `h` 和 `w` 就分别存储了张量 `x` 的高度和宽度。
        return to_4d(self.body(to_3d(x)), h, w)

##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature  #
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        # self.norm1: 第一个层归一化层（Layer Normalization），用于对输入进行归一化。
        # self.attn: 自注意力机制层（Attention），用于计算输入序列中不同位置的依赖关系。
        # self.norm2: 第二个层归一化层，再次对经过自注意力机制处理后的结果进行归一化。
        # self.ffn: 前馈神经网络层（FeedForward Network），用于处理归一化后的结果，执行非线性变换。

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        # forward 方法定义了数据在 TransformerBlock 内部的前向传播过程：
        # 首先通过 Layer Normalization 对输入进行归一化，然后传入自注意力机制模块进行处理，得到输出。
        # 将原始输入和经过注意力机制处理后的输出进行残差连接，并在结果上再次执行 Layer Normalization。
        # 将经过第二个归一化的结果输入前馈神经网络模块，再次得到输出。

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    # 定义了一个神经网络模型，用于将输入图像进行卷积操作并将图像分解成重叠的图像块。这个类的目的是进行图像分块和嵌入表示。
    # __init__ 方法中初始化了一个卷积层 self.proj，用于实现图像分块。
    # forward 方法定义了模型的前向传播过程，将输入 x 经过卷积层处理，并返回处理后的结果 x。
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        # in_c：输入图像的通道数，默认为 3。
        # embed_dim：嵌入维度，默认为 48。
        # bias：是否使用偏置，默认为 False。
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3,
                              stride=1, padding=1, bias=bias)
        # self.proj：使用 nn.Conv2d 创建了一个二维卷积层。这个卷积层用于将输入图像进行卷积操作，实现图像的分块处理和嵌入表示。具体配置如下：
        # 输入通道数为 in_c，输出通道数为 embed_dim。
        # 卷积核大小为 3x3。
        # 步长为 1，填充为 1。
        # 是否使用偏置根据参数 bias 的设置来确定。

    def forward(self, x):
        x = self.proj(x)
        return x
    # forward 方法定义了模型的前向传播过程，接收输入 x，将输入 x 通过 self.proj 卷积层进行处理，
    # 并返回处理后的结果 x。在这个模型中，self.proj 所做的操作可以理解为将输入图像进行卷积处理，以实现图像的分块和嵌入表示。


class Restormer_Encoder(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 dim=64,
                 num_blocks=[4, 4],
                 num_blocks_B=[1],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 depth=2,
                 drop_path=0.1,
                 use_checkpoint=False,
                 norm_layer=nn.LayerNorm,
                 downsample=nn.Identity(),
                 # ===========================
                 d_state=16,
                 dt_rank="auto",
                 ssm_ratio=2.0,
                 attn_drop_rate=0.0,
                 shared_ssm=False,
                 softmax_version=False,
                 # ===========================
                 mlp_ratio=4.0,
                 drop_rate=0.0,
                 **kwargs,
                 ):
        # inp_channels: 输入通道数，默认为 1。
        # out_channels: 输出通道数，默认为 1。
        # dim: 模型的维度，默认为 64。
        # num_blocks: TransformerBlock 的堆叠数量列表，默认为 [4, 4]。
        # heads: 多头注意力机制的头数列表，默认为 [8, 8, 8]。
        # ffn_expansion_factor: 前馈神经网络扩展因子，默认为 2。
        # bias: 是否使用偏置，默认为 False。
        # LayerNorm_type: 层归一化的类型，默认为 'WithBias'。
        super(Restormer_Encoder, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        # self.encoder_level1 = nn.Sequential(
        #     *[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
        #                        bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.encoder_level1 = nn.Sequential(
            *[MBaseFeatureExtraction(hidden_dim=dim,
                                    drop_path=drop_path,
                                    norm_layer=norm_layer,
                                    attn_drop_rate=attn_drop_rate,
                                    d_state=d_state,
                                    dt_rank=dt_rank,
                                    ssm_ratio=ssm_ratio,
                                    shared_ssm=shared_ssm,
                                    softmax_version=softmax_version,
                                    use_checkpoint=use_checkpoint,
                                    mlp_ratio=mlp_ratio,
                                    act_layer=nn.GELU,
                                    drop=drop_rate,
                                    **kwargs,) for i in range(num_blocks_B[0])])
        # nn.Sequential 是 PyTorch 中用于构建序列模型的容器。它接受一系列的层或模块作为输入，并按照顺序依次执行这些层。
        # *[...] 是 Python 中的解包操作符。它将列表或可迭代对象中的每个元素解包成单独的元素。
        # [TransformerBlock(...) for i in range(num_blocks[0])] 是一个列表推导式，用于生成 num_blocks[0] 个 TransformerBlock 的实例，并存储在列表中。

        # 整个代码段通过 nn.Sequential() 将生成的 TransformerBlock 实例按照顺序组成一个神经网络层序列。
        # 这表示 Restormer_Encoder 中的 self.encoder_level1 是一个包含多个 TransformerBlock 的序列，每个 TransformerBlock 都按照指定的参数进行初始化。


        # self.baseFeature = BaseFeatureExtraction(dim=dim, num_heads=heads[2])

        self.MbaseFeature = nn.Sequential(
            *[MBaseFeatureExtraction(hidden_dim=dim,
                                    drop_path=drop_path,
                                    norm_layer=norm_layer,
                                    attn_drop_rate=attn_drop_rate,
                                    d_state=d_state,
                                    dt_rank=dt_rank,
                                    ssm_ratio=ssm_ratio,
                                    shared_ssm=shared_ssm,
                                    softmax_version=softmax_version,
                                    use_checkpoint=use_checkpoint,
                                    mlp_ratio=mlp_ratio,
                                    act_layer=nn.GELU,
                                    drop=drop_rate,
                                    **kwargs,) for i in range(num_blocks[0])])

        self.detailFeature = DetailFeatureExtraction()

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        inp_enc_level1 = inp_enc_level1.permute(0, 2, 3, 1)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        out_enc_level1 = out_enc_level1.permute(0, 3, 1, 2)
        out_enc_level1 = out_enc_level1.permute(0, 2, 3, 1)
        base_feature = self.MbaseFeature(out_enc_level1)
        base_feature = base_feature.permute(0, 3, 1, 2)
        out_enc_level1 = out_enc_level1.permute(0, 3, 1, 2)
        detail_feature = self.detailFeature(out_enc_level1)
        return base_feature, detail_feature, out_enc_level1


class BaseChannelAttention(nn.Module):
    def __init__(self, num_feat, squeeze_factor=16):
        super(BaseChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        # max_out = self.fc(self.max_pool(x))
        # attn = avg_out + max_out
        attn = avg_out
        return x * self.sigmoid(attn)

class DetailChannelAttention(nn.Module):
    def __init__(self, num_feat, squeeze_factor=16):
        super(DetailChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attn = max_out
        return x * self.sigmoid(attn)

class BaseChannelAttentionBlock(nn.Module):

    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super(BaseChannelAttentionBlock, self).__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            BaseChannelAttention(num_feat, squeeze_factor)
            )

    def forward(self, x):
        return self.cab(x)

class DetailChannelAttentionBlock(nn.Module):

    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super(DetailChannelAttentionBlock, self).__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            DetailChannelAttention(num_feat, squeeze_factor)
            )

    def forward(self, x):
        return self.cab(x)

class M2BaseFeatureExtraction(nn.Module):
    def __init__(self,
                 model_clip,
                 inp_channels=1,
                 out_channels=1,
                 dim=64,
                 hidden_dim: int = 0,
                 num_blocks=[4, 4],
                 num_blocks_B=[1],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 # dim=96,
                 depth=2,
                 drop_path=0.1,
                 use_checkpoint=False,
                 norm_layer=nn.LayerNorm,
                 downsample=nn.Identity(),
                 # ===========================
                 d_state=16,
                 dt_rank="auto",
                 ssm_ratio=2.0,
                 attn_drop_rate=0.0,
                 shared_ssm=False,
                 softmax_version=False,
                 # ===========================
                 mlp_ratio=4.0,
                 drop_rate=0.0,
                 **kwargs,
                 ):
        super(M2BaseFeatureExtraction, self).__init__()
        self.MbaseFeature = nn.Sequential(
            *[MBaseFeatureExtraction(hidden_dim=dim,
                                                   drop_path=drop_path,
                                                   norm_layer=norm_layer,
                                                   attn_drop_rate=attn_drop_rate,
                                                   d_state=d_state,
                                                   dt_rank=dt_rank,
                                                   ssm_ratio=ssm_ratio,
                                                   shared_ssm=shared_ssm,
                                                   softmax_version=softmax_version,
                                                   use_checkpoint=use_checkpoint,
                                                   mlp_ratio=mlp_ratio,
                                                   act_layer=nn.GELU,
                                                   drop=drop_rate,
                                                   **kwargs, ) for i in range(num_blocks_B[0])])



        self.scale2 = nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = BaseChannelAttentionBlock(hidden_dim)
        self.norm2 = norm_layer(hidden_dim)

        self.model_clip = model_clip
        self.prompt_guidance_4 = FeatureWiseAffine(in_channels=512, out_channels=64)


    def forward(self, x, text):
        x = x.permute(0, 2, 3, 1)
        x = self.MbaseFeature(x)
        y = self.conv_blk(self.norm2(x).permute(0, 3, 1, 2).contiguous()) + (x * self.scale2).permute(0, 3, 1,
                                                                                                      2).contiguous()

        b = y.shape[0]
        text_features = self.get_text_feature(text.expand(b, -1)).to(y.dtype)
        y = self.prompt_guidance_4(y, text_features)
        return y

    @torch.no_grad()
    def get_text_feature(self, text):
        text_feature = self.model_clip.encode_text(text)
        return text_feature


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=True):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.MLP = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),
            nn.LeakyReLU(),
            nn.Linear(in_channels * 2, out_channels * (1 + self.use_affine_level))
        )

    def forward(self, x, text_embed):
        text_embed = text_embed.unsqueeze(1)
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.MLP(text_embed).view(batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma + beta) * x + beta
        return x

class Restormer_Decoder(nn.Module):
    def __init__(self,
                 model_clip,
                 inp_channels=1,
                 out_channels=1,
                 dim=64,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 depth=2,
                 drop_path=0.1,
                 use_checkpoint=False,
                 norm_layer=nn.LayerNorm,
                 downsample=nn.Identity(),
                 # ===========================
                 d_state=16,
                 dt_rank="auto",
                 ssm_ratio=2.0,
                 attn_drop_rate=0.0,
                 shared_ssm=False,
                 softmax_version=False,
                 # ===========================
                 mlp_ratio=4.0,
                 drop_rate=0.0,
                 **kwargs,
                 # inp_channels=1,
                 # out_channels=1,
                 # dim=64,
                 # num_blocks=[4, 4],
                 # heads=[8, 8, 8],
                 # ffn_expansion_factor=2,
                 # bias=False,
                 # LayerNorm_type='WithBias',
                 ):

        super(Restormer_Decoder, self).__init__()
        self.reduce_channel = nn.Conv2d(int(dim*2), int(dim), kernel_size=1, bias=bias)
        # self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
        #                                     bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.encoder_level2 = nn.Sequential(
            *[MBaseFeatureExtraction(hidden_dim=dim,
                                     drop_path=drop_path,
                                     norm_layer=norm_layer,
                                     attn_drop_rate=attn_drop_rate,
                                     d_state=d_state,
                                     dt_rank=dt_rank,
                                     ssm_ratio=ssm_ratio,
                                     shared_ssm=shared_ssm,
                                     softmax_version=softmax_version,
                                     use_checkpoint=use_checkpoint,
                                     mlp_ratio=mlp_ratio,
                                     act_layer=nn.GELU,
                                     drop=drop_rate,
                                     **kwargs, ) for i in range(num_blocks[1])])

        self.output = nn.Sequential(
            nn.Conv2d(int(dim), int(dim)//2, kernel_size=3,
                      stride=1, padding=1, bias=bias),
            nn.LeakyReLU(),
            nn.Conv2d(int(dim)//2, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=bias),)
        self.sigmoid = nn.Sigmoid()


        self.model_clip = model_clip

        self.prompt_guidance_4 = FeatureWiseAffine(in_channels=512, out_channels=128)

    def forward(self, inp_img, base_feature, detail_feature, text):
        out_enc_level0 = torch.cat((base_feature, detail_feature), dim=1)

        b = out_enc_level0.shape[0]

        text_features = self.get_text_feature(text.expand(b, -1)).to(out_enc_level0.dtype)

        out_enc_level0 = self.prompt_guidance_4(out_enc_level0, text_features)


        out_enc_level0 = self.reduce_channel(out_enc_level0)
        out_enc_level0 = out_enc_level0.permute(0, 2, 3, 1)

        out_enc_level1 = self.encoder_level2(out_enc_level0)

        out_enc_level1 = out_enc_level1.permute(0, 3, 1, 2)

        if inp_img is not None:
            out_enc_level1 = self.output(out_enc_level1) + inp_img
        else:
            out_enc_level1 = self.output(out_enc_level1)
        return self.sigmoid(out_enc_level1), out_enc_level0

    @torch.no_grad()
    def get_text_feature(self, text):
        text_feature = self.model_clip.encode_text(text)
        return text_feature


if __name__ == '__main__':
    height = 128
    width = 128
    window_size = 8
    modelE = Restormer_Encoder().cuda()
    modelD = Restormer_Decoder().cuda()

