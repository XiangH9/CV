# 计算机视觉实验教程 - CLIP

### 梁小丹 中山大学智能工程学院

## 1. 相关背景知识

### 1.1 多模态学习（Multimodal Learning）

多模态学习（Multimodal Learning）指的是从多个不同模态（数据源或表示方式）中学习和理解信息的机器学习方法。每个模态提供关于数据的不同方面或特征，而多模态学习旨在将这些信息结合起来，以获得更全面、更丰富的理解。CLIP 结合了文本、图像两种模态的信息，使用数量惊人的 4 亿图像文本对进行训练，而相比之下，ImageNet 数据集仅包含 120 万张图像。

### 1.2 零样本分类（Zero-Shot Classification）

零样本分类（Zero-Shot Classification）是指模型在未见过某个类别的情况下能够对其进行正确的分类。在CLIP的零样本分类中，给定一个任务描述（例如一句话描述），CLIP能够推断出与该任务描述相关的图像，即使在训练时没有见过这个具体的任务。这是因为CLIP学到了一种通用的图像和文本表示，使得它能够在表示空间中泛化到新的任务。

### 1.3 对比学习（Contrastive Learning）

对比学习是一种无监督学习的方法，通过最大化相似样本（正样本）之间的相似性，最小化不相似样本（负样本）之间的相似性，来学习表示空间。在CLIP的对比学习中，图像和文本被视为两个不同的视图，目标是将相同内容的图像和文本在表示空间中映射到相邻的位置。

## 2. CLIP (Contrastive Language-Image Pre-Training)

论文链接：https://arxiv.org/abs/2103.00020
官方仓库：https://github.com/openai/CLIP

CLIP 模型是由 OpenAI 在《Learning Transferable Visual Models From Natural Language Supervision》一文提出的一个在4 亿图像-文本对上进行训练的神经网络。它可以通过自然语言指示，预测给定图像最相关的文本片段，而无需直接为任务进行优化，类似于 GPT-2 和 3 的零样本能力。CLIP是一个多模态的视觉与语言模型。它可用于图像与文本的相似性比较以及零样本图像分类。CLIP使用类似ViT（Vision Transformer）的Transformer 模型来获取视觉特征，并使用因果语言模型来获取文本特征。然后，文本和视觉特征都被投影到具有相同维度的潜在空间中。投影后的图像和文本特征之间的点积被用作相似度得分。

### 2.1 多模态对比学习预训练

CLIP的对比学习预训练过程旨在通过最大化正样本对的相似性以及最小化负样本对的相似性，从而学习图像和文本的共享表示空间。以下是CLIP对比学习预训练的详细过程：

1. **构建正负样本对：** 对于每个训练样本，CLIP会构建一个正样本对，其中包含一张图像和一个相关联的文本描述。这个描述可以是图像的标签、说明或其他形式的文本。为了构建负样本对，在训练中假设一个批次包含 N 个文本图像对，得到图像嵌入 [I1, I2 … IN] 和文本嵌入[T1, T2 … TN] ，那么除了彼此对应的<I1,T1>, <I2,T2>，... 作为正样本对以外，同一批次内其余的样本 <I1,T2>, <I1,T3>… <Ii,Tj>（其中 i≠j）都可以作为负样本对。
2. **计算相似度得分：** 对于每个正样本对和负样本对，CLIP计算它们在共享表示空间中的相似度得分。这通常通过计算内积或余弦相似度来实现，产生一个 N X N 矩阵。得分的计算反映了图像和文本在表示空间中的相对位置。
3. **定义对比损失函数：** 对比损失函数的目标是最大化正样本对的相似度得分，并最小化负样本对的相似度得分。因此目标是最大化沿对角线的余弦相似度，而非对角线元素的相似性应最小化（例如，I1 图像由 T1 而不是 T2、T2、T3 等描述）。CLIP使用对称交叉熵损失作为其优化目标。 这种类型的损失最小化了图像到文本的方向以及文本到图像的方向。
4. **优化过程：** 使用反向传播和梯度下降等优化算法，对比损失函数被最小化，以更新模型的权重。这个过程在整个数据集上进行多次迭代，直到模型收敛。

通过这样的对比学习过程，CLIP学到了图像和文本的共享表示，使得模型在未见过的任务中也能够理解和泛化。这使得CLIP在零样本分类等任务中表现出色。

下面以微调预训练的 CLIP 模型为例，展示整个训练过程的代码：

（此代码基于https://medium.com/aimonks/a-guide-to-fine-tuning-clip-models-with-custom-data-6c7c0d1416fb，进行了修改和注释）

1. 构造自定义数据集

````python
import clip
class image_title_dataset():
  def __init__(self, list_image_path, list_txt):
    self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")  
    self.image_path = list_image_path  # 图片路径列表
    self.title  = clip.tokenize(list_txt)  # 使用 CLIP 的分词器对文本列表进行分词处理

  def __len__(self):
    return len(self.title)

  def __getitem__(self, idx):
    image = preprocess(Image.open(self.image_path[idx]))  # 使用CLIP的预处理功能对图像进行预处理
    title = self.title[idx]
    return image, title
  
````

2. 模型微调训练

```python
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel

# 加载预训练模型
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# Adam 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)
# 对称交叉熵损失
loss_img = nn.CrossEntropyLoss()  
loss_txt = nn.CrossEntropyLoss()
# 训练
num_epochs = 30
device = 'cuda'
for epoch in range(num_epochs):
    pbar = tqdm(train_dataloader, total=len(train_dataloader))  # 进度条
    for batch in pbar:
        optimizer.zero_grad()  # 清空梯度
        # 加载数据
        images, texts = batch 
        images= images.to(device)
        texts = texts.to(device)
         # 前向过程
        logits_per_image, logits_per_text = model(images, texts) 
        # 计算损失
        ground_truth = torch.arange(len(images),dtype=torch.long,device=device)  # 步长为 1 的等差数列作为 GT，对应对角线坐标
        total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
        # 反向传播
        total_loss.backward()
        optimizer.step()
        pbar.set_description(f"Epoch {epoch}/{num_epochs}, Loss: {total_loss.item():.4f}")
```

### 2.2 Zero-Shot 图像分类

在测试阶段，可以直接将训练好的CLIP用于其他数据集而不需要finetune。和训练阶段类似，首先将需要分类的图像经过编码器得到特征，然后对于目标任务数据集的每一个标签，或者你自己定义的标签，都构造一段对应的文本，如代码中的 “A photo of a dog”，以此类推。然后经过编码器得到文本和图像特征，接着将文本特征与图像特征做内积，内积最大对应的标签就是图像的分类结果。这就完成了目标任务上的 zero-shot 分类。

```python
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

# 加载模型
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 处理文本和图像为模型可以接受的形式
image = Image.open("Your_Image_Path")
inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

# 进行图像分类
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # 图像-文本相似度得分
probs = logits_per_image.softmax(dim=1)  # 可以用softmax来得到标签概率
```

## 3. CLIP 相关应用及扩展工作

### 3.1 CLIPSeg

CLIPSeg在冻结的CLIP模型之上添加了一个基于 Transformer的解码器，用于Zero-shot和one-shot图像分割。在推理时，CLIPSeg 可以根据任意prompt生成图像分割，prompt可以是文本或图像。这种方法使模型能够在只训练一次的情况下完成三个常见的分割任务，这些任务具有不同的挑战：引用表达式分割、zero-shot分割和one-shot分割。

```python
from transformers import AutoProcessor, CLIPSegForImageSegmentation
from PIL import Image
import requests

processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
texts = ["a cat", "a remote", "a blanket"]
inputs = processor(text=texts, images=[image] * len(texts), padding=True, return_tensors="pt")

outputs = model(**inputs)

logits = outputs.logits
print(logits.shape) 
```

### 3.2 BLIP

BLIP 是一个新的VLP框架，可以灵活地转换到视觉语言理解和生成任务。BLIP通过引导字幕有效地利用了嘈杂的网页数据，其中字幕器（captioner）生成合成字幕，而过滤器（ﬁlter）则删除了嘈杂的字幕。作者在广泛的视觉语言任务上获得了最先进的结果，例如图像文本检索 ，图像字幕和VQA。当以zero-shot方式直接转移到视频语言任务时，BLIP还表现出很强的泛化能力。

BLIP 用于图像描述：

```python
from PIL import Image
import requests
from transformers import AutoProcessor, BlipForConditionalGeneration

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
text = "A picture of"

inputs = processor(images=image, text=text, return_tensors="pt")

outputs = model(**inputs)
```

BLIP 用于图像文本检索：

```python
from PIL import Image
import requests
from transformers import AutoProcessor, BlipForImageTextRetrieval

model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
processor = AutoProcessor.from_pretrained("Salesforce/blip-itm-base-coco")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
text = "an image of a cat"

inputs = processor(images=image, text=text, return_tensors="pt")
outputs = model(**inputs)
```

BLIP 用于图像问答：

```python
from PIL import Image
import requests
from transformers import AutoProcessor, BlipForQuestionAnswering

model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

text = "How many cats are in the picture?"
inputs = processor(images=image, text=text, return_tensors="pt")
outputs = model.generate(**inputs)
print(processor.decode(outputs[0], skip_special_tokens=True))
```

### 3.3 其他扩展工作

#### Chinese-CLIP

论文链接：https://arxiv.org/abs/2211.01335

Chinese-CLIP 是 CLIP (Radford et al., 2021) 在大规模中文图像文本对数据集上的实现。 它能够执行跨模态检索，还可以作为零样本图像分类、开放域目标检测等视觉任务的视觉骨干。

#### X-CLIP

论文链接：https://arxiv.org/abs/2208.02816

 X-CLIP 是 CLIP 在视频上的扩展。 该模型由文本编码器、跨帧视觉编码器、多帧集成 Transformer 和视频专用提示生成器组成。

#### BLIP-2

论文链接：https://arxiv.org/abs/2301.12597

BLIP-2 利用冻结的预训练图像编码器和大型语言模型 (LLM)，在它们之间训练轻量级 12 层 Transformer 编码器，从而在各种视觉语言任务上实现最先进的性能。

## 4 实验内容

### 4.1 基础实验

•根据论文和代码理解 CLIP 方法的原理，重点掌握**对比学习**的思想。

•利用 Transformers 库实现 CLIP 的 zero-shot 图像分类推理过程，可以使用不同类别、复杂场景的图片，测试 CLIP 的性能和局限性。

### 4.2 扩展实验

•根据测试的 CLIP 的局限性和实验文档中的代码，自行标注少量的图像文本对，尝试对 CLIP 进行微调训练。

•测试 CLIPSeg，思考它与之前的分割方法在应用场景、效率等方面的差异。

•对 BLIP 进行视觉问答、图像描述等功能的测试。
