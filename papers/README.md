# 论文阅读清单

> 系统性阅读多模态大语言模型领域的关键论文，按主题分类并追踪阅读进度。

**阅读状态说明**：⬜ 未读 / 📖 在读 / ✅ 已读

---

## 一、基础模型

### 语言模型

| 状态 | 论文 | 年份 | 会议/期刊 | 关键词 | 笔记 |
|:----:|------|:----:|:---------:|--------|:----:|
| ⬜ | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | 2017 | NeurIPS | Transformer, Self-Attention | - |
| ⬜ | [Language Models are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | 2019 | - | GPT-2, 无监督预训练 | - |
| ⬜ | [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165) | 2020 | NeurIPS | GPT-3, In-Context Learning | - |
| ⬜ | [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) | 2023 | - | LLaMA, 开源 LLM | - |
| ⬜ | [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288) | 2023 | - | LLaMA-2, RLHF, Chat | - |
| ⬜ | [Mistral 7B](https://arxiv.org/abs/2310.06825) | 2023 | - | 高效架构, Sliding Window Attention | - |

### 视觉模型

| 状态 | 论文 | 年份 | 会议/期刊 | 关键词 | 笔记 |
|:----:|------|:----:|:---------:|--------|:----:|
| ⬜ | [An Image is Worth 16x16 Words (ViT)](https://arxiv.org/abs/2010.11929) | 2020 | ICLR 2021 | Vision Transformer, Patch Embedding | - |
| ⬜ | [Swin Transformer](https://arxiv.org/abs/2103.14030) | 2021 | ICCV | 层级视觉 Transformer, Shifted Window | - |
| ✅ | [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) | 2021 | ICML | 图文对比学习, Zero-Shot | [笔记](clip.md) |
| ⬜ | [Sigmoid Loss for Language Image Pre-Training (SigLIP)](https://arxiv.org/abs/2303.15343) | 2023 | ICCV | Sigmoid 损失, CLIP 改进 | - |
| ⬜ | [EVA-CLIP: Improved Training Techniques for CLIP](https://arxiv.org/abs/2303.15389) | 2023 | - | 大规模视觉表征 | - |

---

## 二、多模态大语言模型

### 经典架构

| 状态 | 论文 | 年份 | 会议/期刊 | 关键词 | 笔记 |
|:----:|------|:----:|:---------:|--------|:----:|
| ⬜ | [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198) | 2022 | NeurIPS | Cross-Attention, Few-Shot | - |
| ⬜ | [BLIP: Bootstrapping Language-Image Pre-training](https://arxiv.org/abs/2201.12086) | 2022 | ICML | 图文预训练, 数据清洗 | - |
| ⬜ | [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597) | 2023 | ICML | Q-Former, 冻结模型桥接 | - |
| ⬜ | [Visual Instruction Tuning (LLaVA)](https://arxiv.org/abs/2304.08485) | 2023 | NeurIPS | 视觉指令微调, Linear Projection | - |
| ⬜ | [Improved Baselines with Visual Instruction Tuning (LLaVA-1.5)](https://arxiv.org/abs/2310.03744) | 2023 | - | MLP 投影, 更强基线 | - |
| ⬜ | [LLaVA-NeXT: Improved Reasoning, OCR, and World Knowledge](https://llava-vl.github.io/blog/2024-01-30-llava-next/) | 2024 | - | 动态高分辨率, AnyRes | - |

### 开源多模态模型

| 状态 | 论文 | 年份 | 会议/期刊 | 关键词 | 笔记 |
|:----:|------|:----:|:---------:|--------|:----:|
| ⬜ | [Qwen-VL: A Versatile Vision-Language Model](https://arxiv.org/abs/2308.12966) | 2023 | - | 多分辨率, 多任务 | - |
| ⬜ | [Qwen2-VL: Enhancing Vision-Language Model's Perception](https://arxiv.org/abs/2409.12191) | 2024 | - | Naive Dynamic Resolution, M-RoPE | - |
| ⬜ | [InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks](https://arxiv.org/abs/2312.14238) | 2023 | CVPR 2024 | 大规模视觉基础模型 | - |
| ⬜ | [InternVL2: Better than the Best](https://arxiv.org/abs/2404.16821) | 2024 | - | 动态分辨率, 渐进式训练 | - |
| ⬜ | [MiniCPM-V: A GPT-4V Level MLLM on Your Phone](https://arxiv.org/abs/2408.01800) | 2024 | - | 端侧部署, 高效多模态 | - |
| ⬜ | [DeepSeek-VL2: Mixture-of-Experts Vision-Language Models](https://arxiv.org/abs/2412.10302) | 2024 | - | MoE, 视觉语言模型 | - |

### 闭源前沿模型（技术报告）

| 状态 | 论文 | 年份 | 关键词 | 笔记 |
|:----:|------|:----:|--------|:----:|
| ⬜ | [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774) | 2023 | 多模态, 涌现能力 | - |
| ⬜ | [GPT-4V(ision) System Card](https://cdn.openai.com/papers/GPTV_System_Card.pdf) | 2023 | 视觉理解, 安全性 | - |
| ⬜ | [Gemini: A Family of Highly Capable Multimodal Models](https://arxiv.org/abs/2312.11805) | 2023 | 原生多模态 | - |
| ⬜ | [Claude 3 Model Card](https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf) | 2024 | 视觉理解, 长上下文 | - |

---

## 三、训练方法

### 对齐与微调

| 状态 | 论文 | 年份 | 会议/期刊 | 关键词 | 笔记 |
|:----:|------|:----:|:---------:|--------|:----:|
| ⬜ | [Training language models to follow instructions with human feedback (InstructGPT)](https://arxiv.org/abs/2203.02155) | 2022 | NeurIPS | RLHF, 指令遵循 | - |
| ⬜ | [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290) | 2023 | NeurIPS | 无需奖励模型的对齐 | - |
| ⬜ | [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) | 2022 | - | RLAIF, 自我改进 | - |
| ⬜ | [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/abs/2212.10560) | 2022 | ACL 2023 | 指令数据自动生成 | - |

### 参数高效微调

| 状态 | 论文 | 年份 | 会议/期刊 | 关键词 | 笔记 |
|:----:|------|:----:|:---------:|--------|:----:|
| ⬜ | [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) | 2021 | ICLR 2022 | 低秩适配, 参数高效 | - |
| ⬜ | [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) | 2023 | NeurIPS | 量化 + LoRA | - |

### 数据工程

| 状态 | 论文 | 年份 | 会议/期刊 | 关键词 | 笔记 |
|:----:|------|:----:|:---------:|--------|:----:|
| ⬜ | [ShareGPT4V: Improving Large Multi-Modal Models with Better Captions](https://arxiv.org/abs/2311.12793) | 2023 | - | 高质量图文数据 | - |
| ⬜ | [The Cauldron: An Open Large-Scale Dataset for Multimodal Instruction Tuning](https://arxiv.org/abs/2401.00369) | 2024 | - | 多模态指令数据集 | - |

---

## 四、前沿主题

### 多模态 Agent

| 状态 | 论文 | 年份 | 关键词 | 笔记 |
|:----:|------|:----:|--------|:----:|
| ⬜ | [Visual ChatGPT](https://arxiv.org/abs/2303.04671) | 2023 | 工具调用, 视觉推理链 | - |
| ⬜ | [CogAgent: A Visual Language Model for GUI Agents](https://arxiv.org/abs/2312.08914) | 2023 | GUI 操控, 高分辨率 | - |

### 视频理解

| 状态 | 论文 | 年份 | 关键词 | 笔记 |
|:----:|------|:----:|--------|:----:|
| ⬜ | [Video-LLaVA: Learning United Visual Representation](https://arxiv.org/abs/2311.10122) | 2023 | 图像-视频统一 | - |
| ⬜ | [LLaVA-Video: Video Understanding with LLM](https://arxiv.org/abs/2501.00103) | 2025 | 长视频, 时序建模 | - |

### 统一生成与理解

| 状态 | 论文 | 年份 | 关键词 | 笔记 |
|:----:|------|:----:|--------|:----:|
| ⬜ | [Emu: Generative Pretraining in Multimodality](https://arxiv.org/abs/2307.05222) | 2023 | 多模态生成 | - |
| ⬜ | [Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation](https://arxiv.org/abs/2410.13848) | 2024 | 解耦视觉编码 | - |

---

## 阅读进度统计

| 类别 | 总数 | ✅ 已读 | 📖 在读 | ⬜ 未读 |
|------|:----:|:------:|:------:|:------:|
| 基础模型 | 11 | 1 | 0 | 10 |
| 多模态模型 | 16 | 0 | 0 | 16 |
| 训练方法 | 8 | 0 | 0 | 8 |
| 前沿主题 | 6 | 0 | 0 | 6 |
| **合计** | **41** | **1** | **0** | **40** |

> 最后更新：2026-02-21
