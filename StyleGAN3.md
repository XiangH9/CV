# 计算机视觉实验教程 - StyleGAN3

### 梁小丹 中山大学智能工程学院

## 1. StyleGAN3的代码实现

此代码用于帮助理解 StyleGAN3 的网络结构，为了方便阅读和学习关键模块，进行了代码的精简化，省略了一部分的超参数等代码，对于 StyleGAN3 的其他部分（如辨别器、训练流程）请参阅官方代码。在阅读此代码时，建议根据论文中的模型图同步进行方便理解。

### 1.1 映射网络

```python
class MappingNetwork(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) 的维度
        w_dim,                      # Intermediate latent (W) 的维度
        num_ws,                     # Intermediate latents 的输出数量
        num_layers      = 2,        # Mapping layers 的数量
        w_avg_beta      = 0.998,    # 用于在训练期间跟踪 W 的移动平均值的衰减
    ):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        # 构建 Mapping layers
        features = [self.z_dim + (self.w_dim if self.c_dim > 0 else 0)] + [self.w_dim] * self.num_layers
        for idx, in_features, out_features in zip(range(num_layers), features[:-1], features[1:]):
            layer = FullyConnectedLayer(in_features, out_features, activation='lrelu')
            setattr(self, f'fc{idx}', layer)
        self.register_buffer('w_avg', torch.zeros([w_dim]))

    def forward(self, z, update_emas=False):
        # 归一化
        x = z.to(torch.float32)
        x = x * (x.square().mean(1, keepdim=True) + 1e-8).rsqrt()
        # Mapping layers 执行
        for idx in range(self.num_layers):
            x = getattr(self, f'fc{idx}')(x)
        # 更新 W 的移动平均值
        if update_emas:
            self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))
        # 扩展到 num_ws
        x = x.unsqueeze(1).repeat([1, self.num_ws, 1])
        return x
```

### 1.2 合成网络

#### 1.2.1 输入处理

```python
class SynthesisInput(torch.nn.Module):
    def __init__(self,
        w_dim,          # Intermediate latent (W) 的维度
        channels,       # 输出的通道数
        size,           # 输出的尺寸
        sampling_rate,  # 输出的采样率
        bandwidth,      # 输出的带宽
    ):
        super().__init__()
        self.w_dim = w_dim
        self.channels = channels
        self.size = np.broadcast_to(np.asarray(size), [2])
        self.sampling_rate = sampling_rate
        self.bandwidth = bandwidth

        # 用于计算 Fourier features 的频率和相位
        freqs = torch.randn([self.channels, 2])
        radii = freqs.square().sum(dim=1, keepdim=True).sqrt()
        freqs /= radii * radii.square().exp().pow(0.25)
        freqs *= bandwidth
        phases = torch.rand([self.channels]) - 0.5

        self.weight = torch.nn.Parameter(torch.randn([self.channels, self.channels]))   # 相当于 1x1 卷积
        self.affine = FullyConnectedLayer(w_dim, 4, weight_init=0, bias_init=[1,0,0,0])  # 预测仿射变换参数
        self.register_buffer('transform', torch.eye(3, 3))
        self.register_buffer('freqs', freqs)
        self.register_buffer('phases', phases)

    def forward(self, w):
        # 加入 batch 维度
        transforms = self.transform.unsqueeze(0) # [batch, row, col]
        freqs = self.freqs.unsqueeze(0) # [batch, channel, xy]
        phases = self.phases.unsqueeze(0) # [batch, channel]
        # 计算仿射变换的参数
        t = self.affine(w) # t = (r_c, r_s, t_x, t_y)
        t = t / t[:, :2].norm(dim=1, keepdim=True) # t' = (r'_c, r'_s, t'_x, t'_y)
        m_r = torch.eye(3, device=w.device).unsqueeze(0).repeat([w.shape[0], 1, 1])
        m_r[:, 0, 0] = t[:, 0]  # r'_c
        m_r[:, 0, 1] = -t[:, 1] # r'_s
        m_r[:, 1, 0] = t[:, 1]  # r'_s
        m_r[:, 1, 1] = t[:, 0]  # r'_c
        m_t = torch.eye(3, device=w.device).unsqueeze(0).repeat([w.shape[0], 1, 1])
        m_t[:, 0, 2] = -t[:, 2] # t'_x
        m_t[:, 1, 2] = -t[:, 3] # t'_y
        transforms = m_r @ m_t @ transforms
        # 计算 Fourier features 的频率和相位
        phases = phases + (freqs @ transforms[:, :2, 2:]).squeeze(2)
        freqs = freqs @ transforms[:, :2, :2]
        # 计算 Fourier features 的振幅
        amplitudes = (1 - (freqs.norm(dim=2) - self.bandwidth) / (self.sampling_rate / 2 - self.bandwidth)).clamp(0, 1)
        # 构建仿射变换采样网格
        theta = torch.eye(2, 3, device=w.device)
        theta[0, 0] = 0.5 * self.size[0] / self.sampling_rate
        theta[1, 1] = 0.5 * self.size[1] / self.sampling_rate
        grids = torch.nn.functional.affine_grid(theta.unsqueeze(0), [1, 1, self.size[1], self.size[0]], align_corners=False)
        # 计算 Fourier features.
        x = (grids.unsqueeze(3) @ freqs.permute(0, 2, 1).unsqueeze(1).unsqueeze(2)).squeeze(3) # [batch, height, width, channel]
        x = x + phases.unsqueeze(1).unsqueeze(2)
        x = torch.sin(x * (np.pi * 2))
        x = x * amplitudes.unsqueeze(1).unsqueeze(2)
        # 应用 1x1 卷积
        weight = self.weight / np.sqrt(self.channels)
        x = x @ weight.t()
        return x.permute(0, 3, 1, 2) # [batch, channel, height, width]
```

#### 1.2.2 调制与解调制

```python
def modulated_conv2d(
    x,                  # Input tensor: [batch_size, in_channels, in_height, in_width]
    w,                  # Weight tensor: [out_channels, in_channels, kernel_height, kernel_width]
    s,                  # Style tensor: [batch_size, in_channels]
    demodulate  = True, # Apply weight demodulation?
    padding     = 0,    # Padding: int or [padH, padW]
):
    out_channels, in_channels, kh, kw = w.shape
    # 预先归一化
    if demodulate:
        w = w * w.square().mean([1,2,3], keepdim=True).rsqrt()
        s = s * s.square().mean().rsqrt()
    # 调制权重
    w = w.unsqueeze(0) # [NOIkk]
    w = w * s.unsqueeze(1).unsqueeze(3).unsqueeze(4) # [NOIkk]
    # 解调权重
    if demodulate:
        dcoefs = (w.square().sum(dim=[2,3,4]) + 1e-8).rsqrt() # [NO]
        w = w * dcoefs.unsqueeze(2).unsqueeze(3).unsqueeze(4) # [NOIkk]
    # 使用分组卷积使得 调制和卷积 作为一个操作进行
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = conv2d_gradfix.conv2d(input=x, weight=w.to(x.dtype), padding=padding, groups=batch_size)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    return x
```
#### 1.2.3 中间层与 ToRGB 层
```python
class SynthesisLayer(torch.nn.Module):
    def __init__(self,
        w_dim,                          # Intermediate latent (W) dimensionality.
        is_torgb,                       # Is this the final ToRGB layer?
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        in_size,                        # Input spatial size: int or [width, height].
        out_size,                       # Output spatial size: int or [width, height].
        conv_kernel         = 3,        # Convolution kernel size. Ignored for final the ToRGB layer.
        filter_size         = 6,        # Low-pass filter size relative to the lower resolution when up/downsampling.
        conv_clamp          = 256,      # Clamp the output to [-X, +X], None = disable clamping.
    ):
        super().__init__()
        # 仿射变换和 modulated conv2d 权重及 LeakRelu 中的偏执
        self.affine = FullyConnectedLayer(self.w_dim, self.in_channels, bias_init=1)
        self.weight = torch.nn.Parameter(torch.randn([self.out_channels, self.in_channels, self.conv_kernel, self.conv_kernel]))
        self.bias = torch.nn.Parameter(torch.zeros([self.out_channels]))
        self.register_buffer('magnitude_ema', torch.ones([]))
        # 上采样层
        self.up_factor = int(np.rint(self.tmp_sampling_rate / self.in_sampling_rate))
        assert self.in_sampling_rate * self.up_factor == self.tmp_sampling_rate
        self.up_taps = filter_size * self.up_factor if self.up_factor > 1 and not self.is_torgb else 1
        self.register_buffer('up_filter', self.design_lowpass_filter(
            numtaps=self.up_taps, cutoff=self.in_cutoff, width=self.in_half_width*2, fs=self.tmp_sampling_rate))
        # 下采样层
        self.down_factor = int(np.rint(self.tmp_sampling_rate / self.out_sampling_rate))
        assert self.out_sampling_rate * self.down_factor == self.tmp_sampling_rate
        self.down_taps = filter_size * self.down_factor if self.down_factor > 1 and not self.is_torgb else 1
        self.down_radial = use_radial_filters and not self.is_critically_sampled
        self.register_buffer('down_filter', self.design_lowpass_filter(
            numtaps=self.down_taps, cutoff=self.out_cutoff, width=self.out_half_width*2, fs=self.tmp_sampling_rate, radial=self.down_radial))

    def forward(self, x, w):
        # 执行仿射变换
        styles = self.affine(w)
        # 执行 modulated conv2d.
        x = modulated_conv2d(x=x.to(dtype), w=self.weight, s=styles, padding=self.conv_kernel-1,
                             demodulate=(not self.is_torgb), input_gain=input_gain)
        # 执行leaky ReLU（集成了上下采样和 clamping）
        gain = 1 if self.is_torgb else np.sqrt(2)
        slope = 1 if self.is_torgb else 0.2
        x = filtered_lrelu.filtered_lrelu(x=x, fu=self.up_filter, fd=self.down_filter, b=self.bias.to(x.dtype),
            up=self.up_factor, down=self.down_factor, padding=self.padding, gain=gain, slope=slope, clamp=self.conv_clamp)
        return x
```

#### 1.2.4 组合成合成网络

```python
class SynthesisNetwork(torch.nn.Module):
    def __init__(self,**layer_kwargs ):
        super().__init__()
        # 构建 Synthesis 输入处理
        self.input = SynthesisInput(
            w_dim=self.w_dim, channels=int(channels[0]), size=int(sizes[0]),
            sampling_rate=sampling_rates[0], bandwidth=cutoffs[0])
        # 构建 Synthesis layers （含 ToRGB）
        self.layer_names = []
        for idx in range(self.num_layers + 1):
            prev = max(idx - 1, 0)
            is_torgb = (idx == self.num_layers)
            is_critically_sampled = (idx >= self.num_layers - self.num_critical)
            use_fp16 = (sampling_rates[idx] * (2 ** self.num_fp16_res) > self.img_resolution)
            layer = SynthesisLayer(w_dim=self.w_dim, is_torgb=is_torgb, **layer_kwargs)
            name = f'L{idx}_{layer.out_size[0]}_{layer.out_channels}'
            setattr(self, name, layer)
            self.layer_names.append(name)

    def forward(self, ws, **layer_kwargs):
        ws = ws.to(torch.float32).unbind(dim=1)
        # 执行 Synthesis 输入处理和 layers
        x = self.input(ws[0])
        for name, w in zip(self.layer_names, ws[1:]):
            x = getattr(self, name)(x, w, **layer_kwargs)
        if self.output_scale != 1:
            x = x * self.output_scale
        return x
```

### 1.3 映射网络+合成网络=生成器

```python
class Generator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        # 合成网络
        self.synthesis = SynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs)
        # 映射网络
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
      	# 映射
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        # 合成
        img = self.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        return img
```


## 2 实验内容

- 根据 StyleGAN3 模型结构图和代码，理解其不同部分及代码实现；（StyleGAN3 的代码集成度较高，相较于其他代码略难理解）
- 尝试根据官方源码（见压缩包stylegan3-main.zip） 运行生成图片或视频的脚本，体验利用 StyleGAN 的图像生成。
- 探索 StyleGAN3 及后续的改进版本，思考 StyleGAN 的局限性和丞待解决的问题。
		如StyleGAN-XL：https://github.com/autonomousvision/stylegan-xl

