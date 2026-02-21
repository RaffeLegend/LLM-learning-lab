# Awesome Multimodal LLM 资源汇总

> 多模态大语言模型领域的优质学习资源集合，持续更新。

---

## 经典课程与教程

### 大学课程

| 课程 | 机构 | 说明 |
|------|------|------|
| [CS231n: Deep Learning for Computer Vision](http://cs231n.stanford.edu/) | Stanford | 计算机视觉深度学习基础，CNN 到 ViT |
| [CS224N: NLP with Deep Learning](http://web.stanford.edu/class/cs224n/) | Stanford | NLP 基础，涵盖 Transformer 与预训练模型 |
| [CS25: Transformers United](https://web.stanford.edu/class/cs25/) | Stanford | Transformer 在各领域的应用专题 |
| [CS324: Large Language Models](https://stanford-cs324.github.io/winter2022/) | Stanford | 大语言模型专题课程 |
| [11-777: Multimodal Machine Learning](https://cmu-multicomp-lab.github.io/mmml-course/fall2023/) | CMU | 多模态机器学习系统性课程 |
| [EECS 498-007: Deep Learning for Computer Vision](https://web.eecs.umich.edu/~justMDavis/courses/EECS498-007/) | UMich | Justin Johnson 主讲，CS231n 进化版 |

### 在线教程与学习路线

| 资源 | 说明 |
|------|------|
| [LLM Course (mlabonne)](https://github.com/mlabonne/llm-course) | 从零到精通 LLM 的完整学习路线，含笔记和代码 |
| [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course) | Transformers 库官方教程，涵盖 NLP 核心任务 |
| [Hugging Face Deep RL Course](https://huggingface.co/learn/deep-rl-course) | 深度强化学习课程，理解 RLHF 的基础 |
| [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/) | 经典图解 Transformer，直觉理解首选 |
| [Lil'Log (Lilian Weng)](https://lilianweng.github.io/) | OpenAI 研究员的技术博客，深入浅出 |
| [Andrej Karpathy - Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) | 从零手写 GPT，理解语言模型本质 |
| [3Blue1Brown - Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) | 优秀的数学直觉可视化 |

---

## 重要开源项目

### 多模态模型

| 项目 | 机构 | 说明 |
|------|------|------|
| [LLaVA](https://github.com/haotian-liu/LLaVA) | UW-Madison | 视觉指令微调范式的开创者，代码清晰易读 |
| [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) | LLaVA Team | LLaVA 的下一代版本，支持动态分辨率和视频 |
| [Qwen-VL / Qwen2-VL](https://github.com/QwenLM/Qwen2-VL) | 阿里云 | 高性能开源多模态模型系列 |
| [InternVL](https://github.com/OpenGVLab/InternVL) | 上海 AI Lab | 可扩展的视觉-语言基础模型 |
| [MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V) | 清华 / 面壁 | 端侧高效多模态模型 |
| [DeepSeek-VL2](https://github.com/deepseek-ai/DeepSeek-VL2) | DeepSeek | 基于 MoE 的视觉语言模型 |
| [Idefics2](https://huggingface.co/HuggingFaceM4/idefics2-8b) | Hugging Face | 开源多模态模型，训练流程透明 |

### 基础设施与工具

| 项目 | 说明 |
|------|------|
| [Transformers](https://github.com/huggingface/transformers) | Hugging Face 核心库，支持主流模型加载与推理 |
| [PEFT](https://github.com/huggingface/peft) | 参数高效微调库（LoRA、QLoRA 等） |
| [TRL](https://github.com/huggingface/trl) | Transformer 强化学习库（RLHF、DPO 训练） |
| [vLLM](https://github.com/vllm-project/vllm) | 高性能 LLM 推理引擎，支持 PagedAttention |
| [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) | 一站式 LLM 微调框架，支持 100+ 模型 |
| [OpenCompass](https://github.com/open-compass/opencompass) | 大模型评测平台，支持多维度基准测试 |
| [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) | 多模态模型统一评测框架 |

### 数据相关

| 项目 | 说明 |
|------|------|
| [ShareGPT4V](https://github.com/InternLM/InternLM-XComposer/tree/main/projects/ShareGPT4V) | 高质量图文描述数据 |
| [LAION-5B](https://laion.ai/blog/laion-5b/) | 大规模开源图文数据集 |
| [DataComp](https://www.datacomp.ai/) | 多模态数据集基准竞赛 |

---

## 知名博客与技术文章

### 个人博客

| 博客 | 作者 | 说明 |
|------|------|------|
| [Lil'Log](https://lilianweng.github.io/) | Lilian Weng (OpenAI) | 系统性技术综述，质量极高 |
| [Jay Alammar's Blog](https://jalammar.github.io/) | Jay Alammar | 可视化讲解 Transformer、BERT、GPT 等 |
| [Sebastian Raschka](https://sebastianraschka.com/blog/) | Sebastian Raschka | LLM 训练、微调实践心得 |
| [Cameron Wolfe](https://cameronrwolfe.substack.com/) | Cameron R. Wolfe | 深入浅出的 LLM/MLLM 论文解读 |
| [Eugene Yan](https://eugeneyan.com/) | Eugene Yan | LLM 应用与工程实践 |

### 重要技术文章

| 文章 | 说明 |
|------|------|
| [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) | Transformer 架构图解 |
| [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) | GPT-2 架构与生成过程图解 |
| [Visual Instruction Tuning 解读 (Lil'Log)](https://lilianweng.github.io/posts/2023-06-23-agent/) | 多模态模型训练方法综述 |
| [A Survey on Multimodal Large Language Models](https://arxiv.org/abs/2306.13549) | MLLM 综述论文，适合入门全局概览 |
| [MM-LLMs: Recent Advances in MultiModal Large Language Models](https://arxiv.org/abs/2401.13601) | 2024 年多模态 LLM 全景综述 |

---

## 基准测试与排行榜

### 综合评测基准

| 基准 | 说明 | 链接 |
|------|------|------|
| MMBench | 多维度多模态能力评测 | [GitHub](https://github.com/open-compass/MMBench) |
| MME | 多模态模型感知与认知评测 | [GitHub](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation) |
| MMMU | 大学级多学科多模态理解 | [官网](https://mmmu-benchmark.github.io/) |
| MathVista | 数学视觉推理 | [官网](https://mathvista.github.io/) |
| SEED-Bench | 多模态大模型生成理解评测 | [GitHub](https://github.com/AILab-CVC/SEED-Bench) |
| RealWorldQA | 真实场景视觉问答 | [Hugging Face](https://huggingface.co/datasets/xai-org/RealworldQA) |

### 排行榜

| 排行榜 | 说明 | 链接 |
|--------|------|------|
| Open VLM Leaderboard | 开源多模态模型综合排行 | [Hugging Face](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard) |
| LMSYS Chatbot Arena | 人类偏好投票排行（含多模态） | [官网](https://chat.lmsys.org/) |
| Open LLM Leaderboard | 开源 LLM 基础能力排行 | [Hugging Face](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) |
| VLMEvalKit Leaderboard | 多模态评测工具包排行 | [GitHub](https://github.com/open-compass/VLMEvalKit) |

---

> 最后更新：2026-02-21
