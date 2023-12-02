# 计算机视觉实验教程 - Swin Transformer for Classification and Detection

### 梁小丹 中山大学智能工程学院

## 1. Swin Transformer 简介

论文链接：https://arxiv.org/abs/2103.14030
官方仓库：https://github.com/microsoft/Swin-Transformer

Swin Transformer是一种作为通用视觉任务的 Backbone 而存在的模型，以替代CNN。 ViT 提出把图片划分 patch，用对待词汇的方式来对待每个 patch，轻松将图片建模成序列， 然后通过编码每个 patch 然后计算两两 patch 之间的 attention，来实现聚合信息，但这样应 对更高清的图片时，由于计算量与图像尺寸成次方关系，会受计算资源掣肘。 

Swin Transformer 相比 ViT 而言，采用了基于偏移窗口的局部自注意力，实现了与图像尺寸呈线性关系的计算量，更好地应对高分辨率输入以及密集预测的视觉任务、


## 2. 预备知识

### 2.1 HuggingFace Transformers
Huggingface Transformers 库提供了数以千计的预训练模型，提供了便于快速下载和使用的API，可以把预训练模型用在个人数据集上微调然后通过 model hub 与社区共享。同时，每个定义的 Python 模块均完全独立，方便修改和快速研究实验。

### 2.2 MMDetection
MMDetection 该工具箱源于 MMDet 团队开发的代码库，该团队于 2018 年赢得了 COCO 检测挑战赛，
1) **模块化设计**：将检测框架分解为不同的组件，通过组合不同的模块可以轻松构建定制的目标检测框架
2) **开箱即用的多任务支持**: 该工具箱直接支持对象检测、实例分割、全景分割和半监督对象检测等多种检测任务
3) **高效率**: 所有基本的 bbox 和 mask 操作都在 GPU 上运行。 训练速度比其他代码库更快或相当，包括 Detectron2、maskrcnn-benchmark 和 SimpleDet


## 3. Swin Transformer for Image Classification (HuggingFace Transformers)

Huggingface Transformers 库提供了便于快速下载和使用的API，方便修改和快速研究实验。
由于我们之前的课程中已经配置了 Pytorch 环境，使用 transformers 只需要简单安装即可：
```
pip install transformers -i https://pypi.tuna.tsinghua.edu.cn/simple
```
安装完成后，如下代码所示，只需要十行左右就可以便捷的实现使用 Swin Transformer 进行图像分类任务
Transformers库会自动从 huggingface hub 上下载模型预训练权重。

```python
from transformers import AutoImageProcessor, SwinForImageClassification
import torch
from PIL import Image

# 将文件路径替换为你的测试图片的路径
image = Image.open("your_test_image.jpg")

# Transformers 库将自动下载模型和处理器的权重
image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
model = SwinForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

# 将图片预处理为模型可接受的格式
inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

# 模型预测的结果为 ImageNet 的 1000 个类别之一
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
```

如果网络无法连接到 huggingface，可以使用国内镜像库手动下载
```shell
# 安装依赖
pip install -U huggingface_hub hf_transfer -i https://pypi.tuna.tsinghua.edu.cn/simple
# 添加镜像
export HF_ENDPOINT=https://hf-mirror.com
# 下载权重及配置文件（请将 local-dir 改为你的本地路径）
huggingface-cli download --resume-download microsoft/swin-tiny-patch4-window7-224 --local-dir YOU_PATH
```

然后将 Python 代码中的模型加载路径修改为你的本地下载路径：
```python
# Transformers 库将自动下载模型和处理器的权重
# [-] image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
image_processor = AutoImageProcessor.from_pretrained("YOU_PATH")
# [-] model = SwinForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
model = SwinForImageClassification.from_pretrained("YOU_PATH")
```

## 4. Swin Transformer for Object Detection (MMDetection)

### 4.1 安装 MMDetection
#### Step 0. Install MMCV & MMEngine using MIM.
```shell
pip install -U openmim
mim install "mmcv>=2.0.0rc4"
mim install "mmengine>=0.7.0"
```
#### Step 1. Install MMDetection.
```shell
pip install mmdet
```
#### Step 2. Check the installation.
为了验证 MMDetection 是否安装正确，首先需要下载配置和检查点文件:
```shell
mim download mmdet --config yolov3_mobilenetv2_320_300e_coco --dest .
```
完成后，将在当前目录中找到两个文件 
1. yolov3_mobilenetv2_320_300e_coco.py 
2. yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth


打开 python 解释器并复制并粘贴以下代码:
```python
from mmdet.apis import init_detector, inference_detector

config_file = 'yolov3_mobilenetv2_320_300e_coco.py'
checkpoint_file = 'yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth'
model = init_detector(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'
inference_detector(model, 'demo/demo.jpg')
```
运行后将看到打印的数组列表，指示检测到的边界框。

### 4.2  Swin Transformer for Object Detection
由于 Swin Transformer 用于目标检测与分割的模型已经集成到 MMDetection 中，因此我们可以十分方便快捷的调用。
```python
from mmdet.apis import DetInferencer
DetInferencer.list_models('mmdet')  # 列出 MMDetection 中所有模型名称
# 这里以 mask-rcnn_swin-s-p4-w7_fpn_amp-ms-crop-3x_coco 为例，MMDet同样可以自动下载模型权重
inferencer = DetInferencer("mask-rcnn_swin-s-p4-w7_fpn_amp-ms-crop-3x_coco")
image_path = "./sysu_cat.jpg"  # 可以将文件路径替换为你想要测试的图片
inferencer(image_path, out_dir='./output')
```

### 5 扩展练习

#### 5.1 Swin Transformer 模型复现
1. 阅读 Swin Transformer 论文（ https://arxiv.org/abs/2103.14030 ）；
2. 尝试根据论文中的细节，使用 Pytorch 框架进行模型复现；
3. 疑难处可以参考官方仓库，比较复现代码与官方代码的差异；

#### 5.2 自定义数据集上微调 Swin Transformer
1. 收集和注释自己的数据集（与第七讲小项目类似）；
2. 利用 MMDet 框架在自定义数据集上进行微调训练（参考仓库 https://github.com/DoraemonTao/Swin-Transformer-Object-Detection ）；
3. 利用之前学习到的指标来评估模型在自定义数据集上的性能；






