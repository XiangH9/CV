# 计算机视觉实验教程 - Vision Transformers

### 梁小丹 中山大学智能工程学院

## 1. Vision Transformer (ViT) 模型结构及实现
论文链接：https://arxiv.org/abs/2010.11929
官方仓库：https://github.com/google-research/vision_transformer

ViT 提出于 《An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale》这篇文章。
受 NLP 中Transformer 模型一系列成功的启发，ViT 尝试将标准 Transformer 模型直接应用于图像，尽可能减少修改。
为此，ViT 将图像分割成块，并将这些块的线性嵌入序列作为Transformer 模型的输入，把图像块与当做 NLP 中的 Token（一般指单词）进行相同的处理。
ViT 以监督的方式训练了图像分类模型，证明了仅仅使用 Transformer 结构也能够在图像分类任务中表现很好。

### 1.1 ViT 模型结构

ViT 模型的主要的创新点在于将图像分块嵌入转化图像嵌入序列来适应 Transformer 模型的输入，随后使用 Transformer Encoder 进行图像特征提取，最后将特征通过 MLP 层后进行分类预测。

#### 1.1.1 图像分块嵌入

在 Transformer 结构中，输入是一个二维的矩阵，矩阵的形状可以表示为$ (𝑁,𝐷)$，其中 $𝑁$ 是序列长度，而$ 𝐷 $是序列中每个向量的维度。因此，在 ViT 算法中，首先需要设法将 $𝐻×𝑊×𝐶 $的三维图像转化为$ (𝑁,𝐷)$ 的二维输入。

ViT中的具体实现方式为：将 $ 𝐻×𝑊×𝐶 $ 的图像，变为一个 $ N×(P^2*C) $ 的序列。这个序列可以看作是一系列展平的图像块，也就是将图像切分成小块后，再将其展平。该序列中一共包含了$ 𝑁=𝐻𝑊/𝑃^2 $个图像块，每个图像块的维度则是$ (𝑃^2∗𝐶)$。其中$ 𝑃 $是图像块的大小，$𝐶 $是通道数量。经过如上变换，就可以将 $𝑁$ 视为序列长度了。

但是，此时每个图像块的维度是$(P^2*C) $  ，而我们实际需要的向量维度是 $𝐷$，因此我们还需要对图像块进行嵌入（Embedding）。这里 Embedding 的方式非常简单，只需要对每个$(P^2*C) $  的图像块做一个线性变换，将维度压缩为 $𝐷 $ 即可。

``` py
from einops import rearrange  # 提供了方便快捷的矩阵形状重新排列的方法
from torch import nn

def to_patch_embedding(img, p, d):
    b, c, h, w = img.shape  # 获取当前 Batch 大小和图像尺寸
    assert h % p == 0 and w % p == 0, f'height {h} or width {w} not divisible by patch size {p}'
    img = rearrange(img, 'b c (h p1) (w p) -> b (h w) (p p c)', p=p)  # 图像分块排列 b × n × p^2*c
    img = nn.Linear(p*p*c, d)(img)  # 通过线性层映射图像块维度到 d,得到 b × n × d
    return img
```

实际模型中，我们还需要在其中加入 `LayerNorm` 来对每个样本的特征进行归一化使其转换为均值为0，方差为1的数据，从而可以避免数据落在激活函数的饱和区，以减少梯度消失的问题。同时，这所有的操作都是顺序执行的，因此我们可以将其封装在一个 `nn.Sequential`中:

```py
from einops.layers.torch import Rearrange
from torch import nn

to_patch_embedding = nn.Sequential(  # nn.Sequential 用于封装按顺序执行的多个模块
    Rearrange('b c (h p1) (w p) -> b (h w) (p p c)', p=p),
    nn.LayerNorm(p),  # 对每个样本的特征进行归一化使其转换为均值为0，方差为1的数据
    nn.Linear(p, d),
    nn.LayerNorm(d),
)
```

为了实现图像分类的工作，ViT 设计了一个可学习的嵌入向量  Class Token，添加到得到的图像嵌入最前面，这样就可以用这个 Class Token 进行分类预测：

```python
cls_token = nn.Parameter(torch.randn(1, 1, d))  # 可学习的 cls_token
img_tokens = to_patch_embedding(img, p, d)  # 图像分块嵌入
b, n, _ = img_tokens.shape
cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)  # 复制以适应 batch 大小
img_tokens = torch.cat((cls_tokens, img_tokens), dim=1)  # 添加到 img_tokens 前
```

但是进行了图像分块嵌入之后，图像的空间维度的信息就丢失了（因为从三维变成了二维）。为了缓解这个问题，ViT 将位置嵌入直接加到图像嵌入以保留位置信息。具体而言，使用标准的可学习1D位置嵌入，在得到图像嵌入后，将其加到每一个图像嵌入中，所得到的嵌入向量序列用作编码器的输入。

```python
pos_embedding = nn.Parameter(torch.randn(1, n + 1, d))  # 可训练位置嵌入
img_tokens += pos_embedding[:, :(n + 1)]  # 对应元素相加
```

我们将以上的操作，整合成为一个 ViT 模型中，可以得到：

```py
# 以下代码仅用于帮助理解，不可直接运行
class ViT(nn.Module):
    def __init__(self, ...):
        super().__init__()
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)  # 图像分块嵌入
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)  # cls_token
        x += self.pos_embedding[:, :(n + 1)]  # 位置嵌入

        x = self.transformer(x)  # Transformer Encoder
        return self.mlp_head(x[:, 0])  # 取 cls_token 进行图像分类
```

### 1.1.2 Transformer Encoder

将图像转化为$ N×(P^2*C) $ 的序列后，就可以将其输入到 Transformer 结构中进行特征提取了，Transformer 模型通常是多个 Transfromer 块的重复。

一个 Transformer 块包括一个 多头自注意力（Multi-Head Self-Attention）层 和一个 MLP 层，每个层前都包含一个 `LayerNorm` 层，同时，两者的头尾还包括一个残差连接（也称跳跃连接）。

多头自注意力是 Transformer 结构中最重要的一个部分，

```py
def multi_head_self_attention(sequence, heads, dim_head):
    b, n, d = sequence.shape

    q, k, v = [
        nn.Linear(d, heads * dim_head, bias=False)(sequence)
        for _ in range(3)
    ]  # 通过线性层得到 Queries, Keys, Values
    q, k, v = [
        rearrange(_, 'b n (h d) -> b h n d', h=heads)
        for _ in (q, k, v)
    ]  # 将 Queries, Keys, Values 按照 heads 进行拆分
		
    # Queries 和 Keys 计算向量内积并归一化
    dots = torch.matmul(q, k.transpose(-1, -2)) / dim_head ** 0.5  
    attention = nn.Softmax(dim=-1)(dots)  # 计算权重得到注意力矩阵
    out = torch.matmul(attention, v)  # 利用注意力矩阵对 Values 加权求和
    out = rearrange(out, 'b h n d -> b n (h d)')  # 将多头的结果拼接回去

    out = nn.Linear(heads * dim_head, d)(out)  # 再通过线性层映射回原来的维度
    return out
```

实际模型中将该过程组装成一个``nn.Module`并加入`LayerNorm`和`Dropout`的操作：

```python
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
          lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv
        )

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
```

Transformer 结构中还有一个重要的结构就是 MLP，即多层感知机，多层感知机由输入层、输出层和至少一层的隐藏层构成。网络中各个隐藏层中神经元可接收相邻前序隐藏层中所有神经元传递而来的信息，经过加工处理后将信息输出给相邻后续隐藏层中所有神经元。在多层感知机中，相邻层所包含的神经元之间通常使用“全连接”方式进行连接。多层感知机可以模拟复杂非线性函数功能，所模拟函数的复杂性取决于网络隐藏层数目和各层中神经元数目。

```python
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
```

有了这些基础的模块，我们就可以组装成最终的 Transformer Encoder：

```python
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x  # 注意此处的残差连接
            x = ff(x) + x  # 注意此处的残差连接
        return self.norm(x)
```

完整代码参考 `vit.py` 文件

## 2. ViT 的其他变体

### 2.1 LeViT

论文地址：https://arxiv.org/abs/2104.01136

项目地址：https://github.com/facebookresearch/LeViT

LeViT 主要优化了 ViT 的计算问题，借鉴了 ResNet 改善 VGG 的思路，即在前两个阶段，以相对较小的计算量应用强分辨率降低。具体而言，在 LeViT 中，作者采用4 个 Conv 层将(3, 224, 224) 的图像降采样到 (256, 14, 14)。

另外，ViT 的类别编码在 LeViT 种被替换为最后激活图的平均池化，卷积层后的 Norm 也使用了初始化为 0 的 BN，可以在推理阶段与卷积层合并以实现加速；同时，在模型的结构上，利用三个阶段去提取特征，每个阶段中间有一个用于降采样的 Attention（实现形式为在 Patch 化之前做一个降采样），使得整个过程是不同尺度下进行的。

### 2.2 CrossViT

论文地址：http://arxiv.org/abs/2103.14899

项目地址：https://github.com/IBM/CrossViT

CrossViT 创新地提出了使用两个具有不同计算复杂度的独立分支来处理小patch和大patch的图像tokens，然后将这些tokens使用交叉注意力多次融合以相互补充。

此外，为了减少计算量，CrossViT 开发了一个简单而有效的基于交叉注意力的token融合模块。具体而言，在每一个分支中，它使用单个token（即 [CLS] token）作为query，与其他分支交换信息，使得其计算复杂度和显存消耗与输入特征大小呈线性关系 。

### 2.3 其他变体

**PVT**

论文地址：https://arxiv.org/abs/2102.12122

金字塔结构的VIT，将transformer融入到CNNs中，在图像的密集分区上进行训练，以实现输出高分辨率。克服了transformer对于密集预测任务的缺点。

**T2T-ViT**

论文地址：https://arxiv.org/abs/2101.11986

通过递归地将相邻的token聚合为一个token，图像最终被逐步结构化为token；提供了具有更深更窄的高效backbone；将图像结构化为token。

**MobileViT**

论文地址：https://arxiv.org/abs/2110.02178

将mobilenet v2连接vit，效果明显优于其他轻量级网络。结合逆残差（inverse residual）和Vit。

更多 ViT 相关的变体模型参考 https://github.com/lucidrains/vit-pytorch

## 3 实验内容

### 3.1 复现ViT

•使用 PyTorch 框架复现整个 ViT 的模型结构。

•输出每一层网络输出矩阵的形状，观察特征在网络中的变化；

•通过调节模型的Patch 大小、token 维度以及输入图像尺寸等，计算模型在不同参数下的 FLOPs 和 Params 。

### 3.2 尝试复现 ViT 变体模型

•根据讲过的 Swin Transformer、 LeViT 、 CrossViT 等变体模型结构，尝试在已经复现的 ViT 的基础上进行修改，复现变体模型；

•在大致相同的超参数的情况下，比较不同模型的 FLOPs 和 Params 以及推理速度等。





