# 多模态大语言模型（MLLM）发展历史全景

> 从早期视觉-语言融合方法到最新统一多模态模型的完整演进时间线

## 关键概念

- **多模态（Multimodal）**：同时处理两种或多种模态（如图像、文字、音频、视频）的能力
- **视觉编码器（Vision Encoder）**：将图像转化为向量表示的模块（如 CNN、ViT）
- **跨模态对齐（Cross-modal Alignment）**：使不同模态的表示在同一语义空间中相互对应
- **对比学习（Contrastive Learning）**：通过拉近正样本对、推远负样本对来学习表示
- **指令微调（Instruction Tuning）**：用人类指令格式的数据对预训练模型进行微调

---

## 详细笔记

### 第一阶段：深度学习之前（~2012 年以前）

#### 核心方法：手工特征 + 统计模型

在深度学习兴起之前，视觉-语言研究依赖手工设计的特征和概率图模型。

**图像标注（Automatic Image Annotation）**

早期研究将图像标注视为一个统计关联问题：给定图像的视觉特征，预测最可能出现的关键词。

- **代表工作**：Barnard et al., 2003，"Matching Words and Pictures"，发表于 *JMLR*
  - 使用 **CorLDA（相关隐狄利克雷分配）**，将图像区域与文字标签在隐含主题空间中建立联系
  - 视觉特征：基于颜色直方图、纹理描述子的手工特征
  - 文字表示：词袋模型（Bag-of-Words, BoW）
  - 核心思想：图像区域和对应词语由同一隐含主题生成

**图像描述的排名框架（Ranking-based Description）**

- **代表工作**：Hodosh et al., 2013，"Framing Image Description as a Ranking Task"，发表于 *JAIR*
  - 提出 **Flickr8k 数据集**：8,000 张图像，每张配有 5 条人工描述句子
  - 将"生成描述"问题转化为"从候选句子中排序找最佳匹配"问题
  - 使用核典型相关分析（KCCA）对齐图像特征和句子特征

**这一时期的局限性**：
- 视觉特征全靠手工设计（SIFT、HOG、颜色直方图），无法捕获高层语义
- 语言模型极其简单（BoW），无法理解句子结构
- 两种模态的表示能力均远不足以支撑复杂的跨模态任务

---

### 第二阶段：CNN + RNN 时代（2014–2017）

#### 转折点：AlexNet 与序列生成

2012 年，AlexNet 在 ImageNet 竞赛上的突破性表现开启了深度学习时代。研究者随即将 CNN 强大的视觉特征提取能力与 RNN 的序列生成能力结合，创造了第一批真正意义上的端到端视觉-语言模型。

---

#### 图像描述生成（Image Captioning）

**Show and Tell（2014）**

- **论文**：Vinyals et al., "Show and Tell: A Neural Image Caption Generator"
- **机构**：Google
- **发表**：arXiv 2014 年 11 月，CVPR 2015
- **作者**：Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan

**架构**（编码器-解码器范式）：

```
图像 → CNN（GoogLeNet/InceptionNet）→ 图像向量
图像向量 → LSTM 初始隐状态
LSTM → 逐词生成描述
```

直觉理解：把 CNN 当"眼睛"，把图像压缩成一个固定长度的向量，再把这个向量"喂"给 LSTM 作为起点，让 LSTM 像写句子一样逐词生成描述。

**局限**：图像信息只在生成开始时传入一次，生成后续词时已"忘记"图像细节。

---

**Show, Attend and Tell（2015）**

- **论文**：Xu et al., "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention"
- **机构**：多伦多大学（Yoshua Bengio 团队）
- **发表**：ICML 2015
- **arXiv**：[1502.03044](https://arxiv.org/abs/1502.03044)
- **作者**：Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhutdinov, Richard Zemel, Yoshua Bengio

**核心创新：视觉注意力机制**

与 Show and Tell 不同，模型在生成每个词时，动态地"看"图像的不同区域：

```
图像 → CNN → 空间特征图（H×W×C）
每一步生成：注意力权重 = f(LSTM 隐状态, 所有空间位置)
           上下文向量 = Σ(注意力权重 × 空间特征)
           当前词 = LSTM(上下文向量, 上一词, 隐状态)
```

提出了两种注意力机制：
- **软注意力（Soft Attention）**：对所有区域加权求和，可微分，用标准反向传播训练
- **硬注意力（Hard Attention）**：随机采样某个区域，用 REINFORCE 算法训练

在 Flickr8k、Flickr30k 和 MS COCO 上均创造了当时最优成绩。

---

**MS COCO 数据集（2015）**

- **论文**：Lin et al., "Microsoft COCO: Common Objects in Context" / "Microsoft COCO Captions"
- **规模**：330,000 张图像，超过 150 万条人工描述
- **意义**：成为后续十年图像描述和视觉问答领域最重要的基准

---

#### 视觉问答（Visual Question Answering, VQA）

**VQA 数据集与任务（2015）**

- **论文**：Antol et al., "VQA: Visual Question Answering"
- **机构**：弗吉尼亚理工、佐治亚理工
- **发表**：ICCV 2015
- **arXiv**：[1505.00468](https://arxiv.org/abs/1505.00468)
- **作者**：Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C. Lawrence Zitnick, Devi Parikh

**数据集规模**：
- 250,000 张图像
- 760,000 个问题
- 10,000,000 条回答（每题 10 个人工回答）

**任务定义**：给定图像 + 自然语言问题 → 预测正确答案（开放式或选择式）

**主流基线架构**：
```
图像 → CNN → 图像特征向量
问题 → LSTM → 问题特征向量
融合（点乘/拼接）→ 分类器 → 答案
```

---

#### 这一时期的架构特征总结

| 组件 | 典型实现 |
|------|---------|
| 视觉编码器 | VGGNet、GoogLeNet、ResNet |
| 语言解码器 | LSTM、GRU |
| 特征融合 | 拼接（Concatenation）、逐元素乘法、双线性融合 |
| 主要任务 | 图像描述、VQA、图像-文字检索 |

---

### 第三阶段：Transformer 与 BERT 风格预训练（2017–2020）

#### 背景：两大变革

1. **Transformer（Vaswani et al., 2017）**："Attention is All You Need"，以自注意力机制替代 RNN
2. **BERT（Devlin et al., 2018）**：双向 Transformer 编码器 + 大规模预训练，横扫 NLP 基准

研究者迅速问：能否把 BERT 的成功经验迁移到多模态？

#### ViT：让 Transformer 直接处理图像

**Vision Transformer（ViT）**

- **论文**：Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
- **机构**：Google Brain
- **发表**：ICLR 2021（arXiv: 2020 年 10 月）
- **arXiv**：[2010.11929](https://arxiv.org/abs/2010.11929)

**核心思想**：把图像切成固定大小的"图块（patches）"，每个 patch 展平后线性映射为向量，然后像处理词序列一样用标准 Transformer 处理。

```
图像（224×224） → 切成 196 个 16×16 的 patch
每个 patch → 线性投影 → 向量（维度 D）
加上位置嵌入（Position Embedding）
→ Transformer 编码器
```

ViT 后来成为多模态模型中视觉编码器的主流选择。

---

#### 多模态 BERT 预训练模型群

这一时期涌现出大量将 BERT 风格预训练扩展到视觉-语言的工作，可分为两种架构路线：

**路线一：双流模型（Two-Stream）**
- 图像和文字各有独立的 Transformer 编码器，通过跨模态注意力层交互

**路线二：单流模型（Single-Stream）**
- 图像特征和文字 token 拼接后，输入同一个 Transformer 编码器联合处理

---

**ViLBERT（2019）— 奠基之作**

- **论文**：Lu et al., "ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks"
- **机构**：佐治亚理工 / Facebook AI Research
- **发表**：NeurIPS 2019
- **arXiv**：[1908.02265](https://arxiv.org/abs/1908.02265)
- **架构**：双流，通过**协同注意力 Transformer 层（Co-attentional Transformer Layers）**交互
- **视觉输入**：来自 Faster R-CNN 检测到的目标区域特征（Region Features），而非全图

**预训练任务**：
1. **多模态掩码建模（Masked Multi-modal Modelling）**：随机掩盖约 15% 的词或图像区域，预测被掩盖内容
2. **多模态对齐预测（Multi-modal Alignment Prediction）**：预测图像和文字段落是否匹配（二分类）

---

**LXMERT（2019）**

- **论文**：Tan & Bansal, "LXMERT: Learning Cross-Modality Encoder Representations from Transformers"
- **机构**：北卡罗来纳大学
- **发表**：EMNLP 2019
- **arXiv**：[1908.07490](https://arxiv.org/abs/1908.07490)
- **架构**：三编码器设计：语言编码器 + 目标关系编码器 + 跨模态编码器（各 9、5、5 层）

**预训练任务**（5 种，最丰富）：
1. 掩码语言建模（MLM）
2. 掩码目标预测（特征回归 + 标签分类）
3. 跨模态匹配（图像-文字是否对应）
4. 视觉问答（直接在预训练中包含 VQA 任务）

**特点**：明确在下游任务（VQA、GQA）相关的数据上预训练，任务导向性强。

---

**VisualBERT（2019）**

- **论文**：Li et al., "VisualBERT: A Simple and Performant Baseline for Vision and Language"
- **机构**：加州大学洛杉矶分校（UCLA）
- **发表**：arXiv 2019，ACL 2020
- **架构**：单流，图像区域特征和文字 token 一起输入 BERT

最简洁的设计：将图像区域特征视为特殊的"视觉词"，与文字 token 共同输入标准 BERT，证明了单流架构的有效性。

---

**UNITER（2020）**

- **论文**：Chen et al., "UNITER: UNiversal Image-TExt Representation Learning"
- **机构**：微软研究院、哥伦比亚大学
- **发表**：ECCV 2020
- **arXiv**：[1909.11740](https://arxiv.org/abs/1909.11740)
- **架构**：单流

**预训练任务**（4 种）：
1. **掩码语言建模（MLM）**：条件掩码，掩盖语言时保留完整图像
2. **掩码区域建模（MRM）**：三变体——特征回归、标签分类、掩码区域分类（KL 散度）
3. **图像-文字匹配（ITM）**
4. **词-区域对齐（WRA）**：使用**最优传输（Optimal Transport）**显式对齐词和图像区域

在 6 个 V+L 任务（9 个数据集）上达到当时最优。

---

#### 这一时期视觉特征的通用做法

几乎所有 BERT 风格多模态模型都使用 Faster R-CNN 提取区域特征：
```
图像 → Faster R-CNN（在 Visual Genome 上预训练）
    → 检测到的目标区域（通常取 36 个）
    → 每个区域：2048 维 ROI 特征 + 4 维位置特征（x1, y1, x2, y2）
```

**瓶颈**：离线目标检测步骤慢、固定区域数量限制了灵活性，后来被端到端 ViT 取代。

---

### 第四阶段：对比学习范式突破（2021）

#### CLIP：用 4 亿图文对对齐视觉与语言

**CLIP（Contrastive Language-Image Pre-Training）**

- **论文**：Radford et al., "Learning Transferable Visual Models From Natural Language Supervision"
- **机构**：OpenAI
- **发布日期**：2021 年 1 月 5 日
- **数据集**：**WIT**（WebImageText）— 从互联网收集的 **4 亿**图文对

**架构**：

```
图像 → 图像编码器（ViT 或 ResNet）→ 图像嵌入向量
文字 → 文字编码器（Transformer）→ 文字嵌入向量

对比学习目标：
  - 批次中 N 对图文，对角线为正样本对
  - 最大化正样本对的余弦相似度
  - 最小化负样本对的余弦相似度（InfoNCE 损失）
```

**零样本分类**：
```
候选类别 "a photo of a {class}" → 文字编码器 → N 个文字向量
测试图像 → 图像编码器 → 图像向量
与所有文字向量计算相似度 → 选最大者为预测类别
```

**关键数字**：
- 训练数据：4 亿图文对（WIT400M）
- 最大模型（ViT-L/14）零样本 ImageNet 准确率：**75.5%**
- 相当于有监督 ResNet-50 的性能，但无需任何 ImageNet 标注数据

**范式转变的意义**：
1. **不再需要人工标注**：用互联网上自然存在的图文对作为训练信号
2. **开放词汇（Open-Vocabulary）**：能识别训练时没见过的类别，只需用文字描述
3. **通用视觉表示**：CLIP 的图像编码器成为后续几乎所有多模态模型的视觉主干

---

#### ALIGN：谷歌用 18 亿嘈杂图文对的探索

**ALIGN（A Large-scale ImaGe and Noisy-text Embedding）**

- **论文**：Jia et al., "Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision"
- **机构**：Google Research
- **发表**：ICML 2021
- **arXiv**：[2102.05918](https://arxiv.org/abs/2102.05918)（首次提交：2021 年 2 月 11 日）
- **数据集**：超过 **18 亿**图像-ALT 文本对（从网络爬取，仅做最小过滤）

**核心思想**：与 CLIP 相似的对比学习架构（EfficientNet 图像编码器 + BERT 文字编码器），但将规模推向极端——即使数据极其嘈杂，足够大的数据量也能产生强大表示。

与 CLIP 的对比：

| | CLIP | ALIGN |
|--|------|-------|
| 数据量 | 4 亿 | 18 亿 |
| 数据过滤 | 较严格（WIT） | 最小过滤 |
| 图像编码器 | ViT / ResNet | EfficientNet |
| 文字编码器 | Transformer | BERT |

---

#### SigLIP：改进对比学习的损失函数（2023）

**SigLIP（Sigmoid Loss for Language-Image Pre-Training）**

- **论文**：Zhai et al., "Sigmoid Loss for Language Image Pre-Training"
- **机构**：Google DeepMind
- **发表**：ICCV 2023
- **arXiv**：[2303.15343](https://arxiv.org/abs/2303.15343)

将 CLIP 的 Softmax 对比损失替换为**逐对 Sigmoid 损失**：不依赖全局批次归一化，支持更小批次更高效训练，性能更优。SigLIP 后来成为 LLaVA、InternVL 等主流多模态模型的首选视觉编码器。

---

### 第五阶段：生成式多模态模型（2022–2023）

这一时期的核心问题：**如何把强大的视觉编码器（如 CLIP）接入预训练 LLM，使其具备视觉理解和图文对话能力？**

---

#### Flamingo：冻结 LLM + 跨注意力接入（2022）

**Flamingo: a Visual Language Model for Few-Shot Learning**

- **机构**：DeepMind（现 Google DeepMind）
- **发表**：NeurIPS 2022
- **arXiv**：[2204.14198](https://arxiv.org/abs/2204.14198)（发布于 2022 年 4 月 29 日）
- **作者**：Jean-Baptiste Alayrac et al.（27 位作者）
- **模型规模**：最大 80B 参数（基于 Chinchilla 70B LLM）

**三大核心架构创新**：

1. **Perceiver Resampler（感知重采样器）**：
   - 将视觉编码器输出的可变数量空间特征，压缩为**固定数量**的视觉 token（通常 64 个）
   - 使用一组可学习查询向量（Latent Queries）通过交叉注意力从视觉特征中提炼信息
   - 意义：无论图像分辨率或视频帧数如何变化，都能得到固定长度的视觉表示

2. **冻结 LLM + 新增跨模态注意力层**：
   - 原始 LLM 权重**完全冻结**，保留其语言能力
   - 在每个冻结的 LLM 层之间插入新的**交叉注意力层（Cross-Attention Layer）**
   - 交叉注意力：Query 来自语言 token，Key/Value 来自 Perceiver Resampler 的视觉输出
   - 使用 tanh 门控机制（初始化为 0）确保训练初期不破坏 LLM 的语言能力

3. **交错图文序列**：
   - 天然支持在一个 prompt 中交错多张图像和文字
   - 配合少样本学习（Few-Shot Learning）：提供几个"图像+问题+答案"示例，然后给测试图像

```
架构示意：
视觉编码器（冻结）→ Perceiver Resampler → 视觉 token
                                            ↓
文字 → [冻结 LLM 层] → [跨注意力层] → [冻结 LLM 层] → [跨注意力层] → 输出
```

**历史意义**：Flamingo 展示了"冻结大模型 + 轻量级接口"的可行性，为后续工作奠定了设计蓝图。

---

#### BLIP：统一理解与生成（2022）

**BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation**

- **机构**：Salesforce Research
- **发表**：ICML 2022
- **arXiv**：[2201.12086](https://arxiv.org/abs/2201.12086)
- **作者**：Junnan Li et al.

**核心创新：MED 架构（Multimodal mixture of Encoder-Decoder）**

BLIP 用**一套共享权重的 Transformer**同时完成三种功能，通过不同的注意力掩码切换模式：
1. **单模态编码器**（BERT 风格）：图文各自编码，用于对比学习
2. **图像引导文字编码器**：图像作为条件，对文字进行编码，用于图文匹配
3. **图像引导文字解码器**：自回归生成文字描述

**CapFilt（标题过滤器）**：解决网络图文数据噪声问题的关键创新：
- 用 BLIP 训练一个**标题生成器**，为图像生成合成标题
- 用 BLIP 训练一个**过滤器**，去除低质量/不相关的原始网络标题
- 混合高质量合成标题 + 过滤后真实标题，迭代训练

---

#### BLIP-2：Q-Former 桥接冻结视觉编码器与冻结 LLM（2023）

**BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models**

- **机构**：Salesforce Research
- **发表**：ICML 2023
- **arXiv**：[2301.12597](https://arxiv.org/abs/2301.12597)（发布于 2023 年 1 月）
- **作者**：Junnan Li et al.

**Q-Former（Querying Transformer）：模态桥梁**

```
冻结视觉编码器（ViT）→ 视觉特征
                          ↓
              Q-Former（可训练，~188M 参数）
              ├── 一组可学习查询向量（32 个）
              ├── 自注意力（查询之间互相交互）
              └── 交叉注意力（查询向视觉特征提问）
                          ↓
              32 个查询向量的输出（固定维度）
                          ↓
              线性投影层
                          ↓
              冻结 LLM（OPT / FlanT5）
```

**两阶段训练**：
- **阶段一**：Q-Former 与冻结视觉编码器对齐（使用三个预训练任务：图像-文字对比、图像-文字匹配、图像引导文字生成）
- **阶段二**：将 Q-Former 输出通过线性层接入冻结 LLM，做生成式视觉-语言预训练

**高效性**：在 11B 参数的 LLM 上，仅训练约 **188M 参数**（< 2%），即可达到当时最优性能。

---

#### LLaVA：视觉指令微调范式（2023）

**LLaVA: Large Language and Vision Assistant**

**论文**：Liu et al., "Visual Instruction Tuning"
- **机构**：威斯康星大学麦迪逊分校、微软研究院、哥伦比亚大学
- **发表**：NeurIPS 2023 Oral
- **arXiv**：[2304.08485](https://arxiv.org/abs/2304.08485)（发布于 2023 年 4 月 17 日）
- **作者**：Haotian Liu, Chunyuan Li, Qingyang Wu, Yong Jae Lee

**核心贡献：用 GPT-4 生成多模态指令数据**

1. 无法直接用 GPT-4 处理图像（当时 GPT-4 只有文字能力），所以采用间接方法：
   - 将图像的**描述文字**和**目标框标注**作为文字输入给 GPT-4
   - GPT-4 根据这些文字信息，生成多模态对话格式的指令数据

2. 生成三类数据：
   - 对话式（Conversation）
   - 详细描述（Detailed Description）
   - 复杂推理（Complex Reasoning）

**架构**（极简设计）：

```
图像 → CLIP 视觉编码器（冻结）→ 视觉特征
                                     ↓
                        线性投影层（可训练）
                                     ↓
文字指令 → [视觉 token, 文字 token] → LLaMA（可微调）→ 回答
```

**两阶段训练**：
- **阶段一**：只训练线性投影层，对齐视觉-语言特征空间（约 60 万条图文对）
- **阶段二**：联合微调线性层和 LLM，学习多模态指令跟随（约 15 万条指令数据）

**成绩**：在合成多模态指令测试集上，相对 GPT-4 达到 85.1% 的得分；在 ScienceQA 上达到 92.53% SOTA。

**历史意义**：LLaVA 开创了"视觉指令微调（Visual Instruction Tuning）"范式，其简洁的架构和开源策略推动了整个领域的快速发展。

---

### 第六阶段：规模扩张时代（2023–2024）

#### 闭源前沿模型

**GPT-4V（OpenAI）**

- **发布**：2023 年 9 月 25 日
- **能力**：接受图像 + 文字输入，生成文字输出
- **技术细节**：系统卡（System Card）公开，但训练细节保密
- **意义**：第一个被广泛商用部署的强大多模态模型，证明了多模态 LLM 的实用价值

**Gemini（Google DeepMind）**

- **发布**：2023 年 12 月 13 日
- **特点**：从头设计为原生多模态（Native Multimodal），而非在纯文字 LLM 上"附加"视觉能力
- **模型系列**：Gemini Ultra（最强）/ Gemini Pro / Gemini Nano
- **宣称**：在 MMLU 上首次超越人类专家水平；支持文字、图像、视频、音频、代码

**Gemini 1.5 Pro（2024 年 3 月）**

- **arXiv**：[2403.05530](https://arxiv.org/abs/2403.05530)
- **核心突破**：上下文窗口扩展至 **100 万 token**（实验性支持 1000 万 token）
- **多模态长上下文**：可处理 1 小时视频、11 小时音频、3 万行代码

**GPT-4o（OpenAI，2024 年 5 月）**

- **发布**：2024 年 5 月
- **"o" = Omni（全能）**：真正原生多模态，端到端处理文字、图像、音频
- **实时性**：对音频输入的响应延迟最低 232ms（平均 320ms），接近人类对话
- **意义**：不同于 GPT-4V（在 LLM 上附加视觉），GPT-4o 是统一处理所有模态的模型

---

#### 开源生态系统

**Qwen-VL（阿里巴巴，2023）**

- **论文**：Bai et al., "Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond"
- **arXiv**：[2308.12966](https://arxiv.org/abs/2308.12966)（2023 年 8 月）
- **模型大小**：7B 参数（基于 Qwen-7B LLM）
- **特色能力**：中英双语、精细定位（输出目标框坐标）、文字识别（OCR）
- **三阶段训练**：预训练 → 多任务预训练 → 指令微调

**InternVL（上海 AI Lab / 清华，2023–2024）**

- **论文**：Chen et al., "InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks"
- **arXiv**：2312.14238（2023 年 12 月）
- **发表**：CVPR 2024 Oral
- **核心创新**：将视觉编码器（InternViT）扩展到 **6B 参数**（与 LLM 同量级），再通过 MLP 连接 LLM
- **架构**：InternViT-6B + MLP 投影层 + 各种规模 LLM（7B 到 34B）
- **后续版本**：InternVL 1.5（2024 年 4 月）、InternVL 2（2024 年）、InternVL 2.5（2024 年 12 月）

**LLaVA-NeXT（2024 年 1 月）**

- **发布**：2024 年 1 月 30 日（[博客](https://llava-vl.github.io/blog/2024-01-30-llava-next/)）
- **核心改进：AnyRes 动态分辨率**
  - 将高分辨率图像切分为多个 336×336 的局部图块，外加一个全图缩略图
  - 支持多种宽高比（最高 672×672、336×1344、1344×336 等）
  - 有效分辨率提升 4×，显著降低幻觉，提升 OCR 和细节识别能力
- **训练效率**：仅用 32 个 GPU 训练约 1 天，成本比同类模型低 100-1000×

**DeepSeek-VL（DeepSeek AI，2024）**

- **论文**：Lu et al., "DeepSeek-VL: Towards Real-World Vision-Language Understanding"
- **arXiv**：[2403.05525](https://arxiv.org/abs/2403.05525)（2024 年 3 月）
- **架构**：混合视觉编码器（低分辨率 SigLIP-L + 高分辨率 SAM 编码器）+ 两层 MLP + DeepSeek LLM
- **特色**：专注真实世界场景（网页截图、PDF、图表、OCR），数据多样性强

**MiniCPM-V（清华大学 / OpenBMB，2024）**

- **论文**：arXiv: [2408.01800](https://arxiv.org/abs/2408.01800)
- **定位**：面向移动端部署的高效多模态模型（"GPT-4V Level MLLM on Your Phone"）
- **技术特点**：高分辨率（任意宽高比）+ 低幻觉（RLAIF-V 方法）+ 多语言（30+ 种语言）+ 8B 参数可在手机运行
- **相关论文发表于** Nature Communications（2025）

---

#### 关键技术趋势总结

**动态分辨率（Dynamic Resolution）**

从固定 224×224 输入演进到支持任意分辨率：
- 主流方法：将图像切分为多个固定大小的块（AnyRes / LLaVA-NeXT 方法）
- 优点：保留更多细节，提升 OCR 和精细理解能力

**视频理解（Video Understanding）**

将视频均匀采样帧后，用图像编码器逐帧处理：
- 挑战：帧数多导致 token 数量爆炸（每帧 256+ token，1 分钟视频可能有数千 token）
- 解决方向：帧压缩（时序池化、Query-based 压缩）+ 长上下文 LLM

---

### 第七阶段：前沿探索（2024–2025）

#### 多模态智能体（Multimodal Agents）

**CogAgent（清华大学 / 智谱 AI，2023）**

- **论文**：Hong et al., "CogAgent: A Visual Language Model for GUI Agents"
- **arXiv**：[2312.08914](https://arxiv.org/abs/2312.08914)（2023 年 12 月）
- **发表**：CVPR 2024
- **模型**：18B 参数（11B 视觉参数 + 7B 语言参数）
- **核心创新**：**双分辨率编码器**——低分辨率编码器（224×224）理解全局语义，高分辨率交叉模块（1120×1120）处理 GUI 细节（小按钮、文字）
- **能力**：仅用屏幕截图即可执行网页和安卓 GUI 操作，性能超越依赖 HTML 文字的 LLM 方法

**GUI 智能体研究方向**：
- 输入：屏幕截图
- 输出：操作指令（点击坐标、输入文字、滚动等）
- 应用：自动化 UI 操作、无障碍访问

---

#### 统一生成+理解架构（Unified Generation & Understanding）

**Emu 系列（BAAI，北京人工智能研究院）**

| 版本 | 时间 | 参数 | 特点 |
|------|------|------|------|
| Emu（2023 年 7 月） | arXiv: [2307.05222](https://arxiv.org/abs/2307.05222) | - | 交错图文序列的统一自回归生成 |
| Emu2（2023 年 12 月） | - | 37B | 最大开源多模态生成模型，强大上下文学习 |
| Emu3（2024 年 10 月） | - | - | 纯 next-token prediction，无扩散模型 |

**NExT-GPT（2023–2024）**

- **论文**：arXiv: [2309.05519](https://arxiv.org/abs/2309.05519)
- **发表**：ICML 2024
- **架构**：LLM + 多模态编码器（ImageBind）+ 多扩散解码器（Stable Diffusion / AudioLDM / Zeroscope）
- **能力**：任意到任意（Any-to-Any）生成——输入和输出均可为文字、图像、视频、音频的任意组合
- **训练技巧**：仅训练 1% 的参数（投影层），大量利用预训练模型

---

**Chameleon：早融合 Token 化统一模型（Meta FAIR，2024）**

- **论文**：Team at Meta, "Chameleon: Mixed-Modal Early-Fusion Foundation Models"
- **arXiv**：[2405.09818](https://arxiv.org/abs/2405.09818)（2024 年 5 月）
- **核心思想**：**早融合（Early Fusion）**——图像和文字使用**统一的 Token 词表**，用同一个 Transformer 从头处理

```
图像 → VQ-VAE Tokenizer → 离散图像 token（如 1024 个 token）
文字 → BPE Tokenizer → 文字 token
两种 token 混合 → 统一词表 → 标准 Transformer（自回归）
```

与传统的"晚融合（Late Fusion）"（各模态独立编码再融合）对比：
- 早融合：更彻底的统一，模型从最底层就处理多模态
- 挑战：联合优化图像 token 和文字 token 的表示空间很困难，需要特殊训练稳定技巧
- 训练规模：7B 和 34B 参数，训练于 4.4 万亿 token

---

**Janus：解耦视觉编码的统一框架（DeepSeek，2024）**

- **论文**："Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation"
- **arXiv**：[2410.13848](https://arxiv.org/abs/2410.13848)（2024 年 10 月）
- **发表**：CVPR 2025

**核心洞见**：多模态**理解**和**生成**对视觉表示的需求不同：
- **理解**需要高层语义特征（语义丰富，如 SigLIP 编码的特征）
- **生成**需要低层像素级细节（结构保留，如 VQ 离散 token）

**架构**：
```
理解路径：图像 → SigLIP 语义编码器 → 理解适配器 → LLM
生成路径：图像 → VQ Tokenizer → 离散图像 token → LLM
两路径共享同一个 Transformer（统一自回归框架）
```

**Janus-Pro（2025 年 1 月）**：在数据和模型规模上进一步扩展，发布于 arXiv: [2501.17811](https://arxiv.org/abs/2501.17811)

---

#### 原生多模态 vs 模块化方法的对比

```
模块化方法（Modular）：         原生多模态（Native Multimodal）：
CLIP 视觉编码器                 统一词表
    ↓ (投影/Q-Former)          （图像 token + 文字 token）
冻结/微调 LLM                       ↓
                              同一个 Transformer
                               处理所有 token
优点：                         优点：
- 训练效率高                   - 深度模态融合
- 可复用已有 LLM               - 天然支持任意序列
- 易于工程化                   - 从底层学习跨模态关系

缺点：                         缺点：
- 模态间存在信息瓶颈            - 训练成本极高
- 图像/语言表示空间有差距       - 需要 VQ 图像 tokenizer
```

---

## 完整演进时间线

```
1990s-2011  手工特征时代
            BoW + 手工视觉特征 → 图像标注/检索

2012        AlexNet，深度学习革命开始

2014-2015   CNN + RNN 时代
            Show and Tell (2014) | Show Attend and Tell (2015)
            VQA 数据集 (2015) | MS COCO Captions (2015)

2017-2020   Transformer + BERT 多模态预训练时代
            ViT (2020) | ViLBERT (2019) | LXMERT (2019)
            VisualBERT (2019) | UNITER (2020)

2021        对比学习范式
            CLIP (Jan 2021) | ALIGN (Feb 2021, ICML 2021)

2022        生成式多模态模型兴起
            BLIP (Jan 2022) | Flamingo (Apr 2022)

2023        指令微调 + 开源生态爆发
            BLIP-2 (Jan 2023) | LLaVA (Apr 2023) | InstructBLIP
            GPT-4V (Sep 2023) | Gemini (Dec 2023)
            Qwen-VL (Aug 2023) | InternVL (Dec 2023)
            SigLIP (2023) | CogAgent (Dec 2023)

2024        规模扩张 + 原生多模态
            LLaVA-NeXT (Jan 2024) | DeepSeek-VL (Mar 2024)
            GPT-4o (May 2024) | Gemini 1.5 (Mar 2024)
            Chameleon (May 2024) | Janus (Oct 2024)
            InternVL 2.5 (Dec 2024)

2025        统一多模态 + 智能体
            Janus-Pro (Jan 2025) | Qwen2.5-VL
            多模态 Agent 持续发展中...
```

---

## 个人理解与思考

### 三条核心主线

回顾整个发展历史，可以归纳出三条并行推进的主线：

1. **视觉表示的演进**：手工特征 → CNN 特征 → 目标检测区域特征 → ViT Patch 特征 → CLIP/SigLIP 对齐特征
   - 趋势：越来越通用、越来越不依赖任务特定监督

2. **连接视觉与语言的方式演进**：线性层投影 → 交叉注意力 → Q-Former → 统一 Token
   - 趋势：从简单到复杂，又回归简单（Token 统一）

3. **训练范式演进**：任务特定训练 → 无监督预训练 → 指令微调 → RLHF/DPO
   - 趋势：越来越通用，越来越对齐人类意图

### 值得深思的问题

1. **图像应该用 Token 表示还是连续特征？**
   - Token（Chameleon、Emu3）：统一框架，天然支持生成
   - 连续特征（LLaVA 系列）：信息损失少，理解性能更好
   - 目前尚无定论，这是 2024-2025 年的核心研究问题

2. **视觉编码器应该多大？**
   - InternVL 将 ViT 扩展到 6B，认为视觉编码器太小是瓶颈
   - LLaVA 用 CLIP ViT-L/14（约 300M），效果也很好
   - 规模定律（Scaling Law）在视觉模态上如何体现？

3. **多模态能力是涌现出来的还是需要刻意训练的？**
   - GPT-4V 的出现暗示：足够大的纯语言模型 + 视觉对齐可能就足够
   - 但专门的多模态预训练（如 Flamingo、Gemini）往往效果更好

---

## 相关链接

### 关键论文

- **Show and Tell**: [Semantic Scholar](https://www.semanticscholar.org/paper/Show-and-tell:-A-neural-image-caption-generator-Vinyals-Toshev/d4dc1012d780e8e2547237eb5a6dc7b1bf47d2f0)
- **Show Attend and Tell**: [arXiv 1502.03044](https://arxiv.org/abs/1502.03044) | [ICML 2015](https://proceedings.mlr.press/v37/xuc15.html)
- **VQA**: [arXiv 1505.00468](https://arxiv.org/abs/1505.00468) | [ICCV 2015](https://openaccess.thecvf.com/content_iccv_2015/html/Antol_VQA_Visual_Question_ICCV_2015_paper.html)
- **ViT**: [arXiv 2010.11929](https://arxiv.org/abs/2010.11929)
- **ViLBERT**: [arXiv 1908.02265](https://arxiv.org/abs/1908.02265) | [NeurIPS 2019](https://proceedings.neurips.cc/paper_files/paper/2019/hash/c74d97b01eae257e44aa9d5bade97baf-Abstract.html)
- **LXMERT**: [arXiv 1908.07490](https://arxiv.org/abs/1908.07490) | [EMNLP 2019](https://aclanthology.org/D19-1514/)
- **UNITER**: [arXiv 1909.11740](https://arxiv.org/abs/1909.11740) | [ECCV 2020](https://link.springer.com/chapter/10.1007/978-3-030-58577-8_7)
- **CLIP**: [OpenAI Blog](https://openai.com/index/clip/) | [GitHub](https://github.com/openai/CLIP)
- **ALIGN**: [arXiv 2102.05918](https://arxiv.org/abs/2102.05918) | [ICML 2021](http://proceedings.mlr.press/v139/jia21b/jia21b.pdf)
- **SigLIP**: [arXiv 2303.15343](https://arxiv.org/abs/2303.15343) | [ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Zhai_Sigmoid_Loss_for_Language_Image_Pre-Training_ICCV_2023_paper.html)
- **BLIP**: [arXiv 2201.12086](https://arxiv.org/abs/2201.12086)
- **Flamingo**: [arXiv 2204.14198](https://arxiv.org/abs/2204.14198) | [NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/file/960a172bc7fbf0177ccccbb411a7d800-Paper-Conference.pdf)
- **BLIP-2**: [arXiv 2301.12597](https://arxiv.org/abs/2301.12597)
- **LLaVA**: [arXiv 2304.08485](https://arxiv.org/abs/2304.08485) | [NeurIPS 2023](https://papers.nips.cc/paper_files/paper/2023/hash/6dcf277ea32ce3288914faf369fe6de0-Abstract-Conference.html)
- **GPT-4V System Card**: [OpenAI](https://cdn.openai.com/papers/GPTV_System_Card.pdf)
- **Gemini 1.5**: [arXiv 2403.05530](https://arxiv.org/abs/2403.05530)
- **Qwen-VL**: [arXiv 2308.12966](https://arxiv.org/abs/2308.12966)
- **InternVL**: [GitHub](https://github.com/OpenGVLab/InternVL)
- **DeepSeek-VL**: [arXiv 2403.05525](https://arxiv.org/abs/2403.05525)
- **MiniCPM-V**: [arXiv 2408.01800](https://arxiv.org/abs/2408.01800)
- **CogAgent**: [arXiv 2312.08914](https://arxiv.org/abs/2312.08914) | [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Hong_CogAgent_A_Visual_Language_Model_for_GUI_Agents_CVPR_2024_paper.pdf)
- **Chameleon**: [arXiv 2405.09818](https://arxiv.org/abs/2405.09818)
- **Janus**: [arXiv 2410.13848](https://arxiv.org/abs/2410.13848)
- **NExT-GPT**: [arXiv 2309.05519](https://arxiv.org/abs/2309.05519)
- **Emu**: [arXiv 2307.05222](https://arxiv.org/abs/2307.05222) | [GitHub](https://github.com/baaivision/Emu)

### 推荐综述

- HuggingFace 博客：[A Dive into Vision-Language Models](https://huggingface.co/blog/vision_language_pretraining)
- HuggingFace 博客：[Vision Language Models Explained](https://huggingface.co/blog/vlms)
- Multimodal Pretraining Unmasked（TACL 2021）：[MIT Press](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00408/107279/)

---

## 更新日志

- 2026-02-21: 初始创建，覆盖深度学习之前至 2025 年初的完整演进历史
