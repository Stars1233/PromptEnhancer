<div align="center">

# PromptEnhancer: A Simple Approach to Enhance Text-to-Image Models via Chain-of-Thought Prompt Rewriting

[**Linqing Wang**](https://scholar.google.com/citations?hl=en&view_op=list_works&gmla=AH8HC4z9rmDHYjp5o28xKk8U4ddD_n7BuMnk8UZFP-jygFBtHUSz6pf-5FP32B_yKMpRU9VpDY3iT8eM0zORHA&user=Hy12lcEAAAAJ) · 
[**Ximing Xing**](https://ximinng.github.io/) · 
[**Yiji Cheng**](https://scholar.google.com/citations?user=Plo8ZSYAAAAJ&hl=en) · 
Zhiyuan Zhao · 
Donghao Li ·
Tiankai Hang ·
Zhenxi Li ·
[**Jiale Tao**](https://scholar.google.com/citations?user=WF5DPWkAAAAJ&hl=en) · 
[**QiXun Wang**](https://github.com/wangqixun) · 
[**Ruihuang Li**](https://scholar.google.com/citations?user=8CfyOtQAAAAJ&hl=en) · 
Comi Chen ·
Xin Li · 
[**Mingrui Wu**](https://scholar.google.com/citations?user=sbCKwnYAAAAJ&hl=en) · 
Xinchi Deng · 
Shuyang Gu · 
[**Chunyu Wang**](https://scholar.google.com/citations?user=VXQV5xwAAAAJ&hl=en)<sup>†</sup> · 
[**Qinglin Lu**](https://luqinglin.weebly.com/)<sup>*</sup>

Tencent Hunyuan

<sup>†</sup>Project Lead · <sup>*</sup>Corresponding Author

</div>

<p align="center">
  <a href="https://www.arxiv.org/abs/2509.04545"><img src="https://img.shields.io/badge/Paper-arXiv:2509.04545-red?logo=arxiv" alt="arXiv"></a>
  <a href="https://zhuanlan.zhihu.com/p/1949013083109459515"><img src="https://img.shields.io/badge/知乎-技术解读-0084ff?logo=zhihu" alt="Zhihu"></a>
  <a href="https://huggingface.co/tencent/HunyuanImage-2.1/tree/main/reprompt"><img src="https://img.shields.io/badge/Model-PromptEnhancer_7B-blue?logo=huggingface" alt="HuggingFace Model"></a>
  <a href="https://huggingface.co/PromptEnhancer/PromptEnhancer-Img2img-Edit"><img src="https://img.shields.io/badge/Model-PromptEnhancer_Img2Img_Edit-blue?logo=huggingface" alt="HuggingFace Model"></a>
  <!-- <a href="https://huggingface.co/PromptEnhancer/PromptEnhancer-32B"><img src="https://img.shields.io/badge/Model-PromptEnhancer_32B-blue?logo=huggingface" alt="HuggingFace Model"></a> -->
  <a href="https://huggingface.co/datasets/PromptEnhancer/T2I-Keypoints-Eval"><img src="https://img.shields.io/badge/Benchmark-T2I_Keypoints_Eval-blue?logo=huggingface" alt="T2I-Keypoints-Eval Dataset"></a>
  <a href="https://hunyuan-promptenhancer.github.io/"><img src="https://img.shields.io/badge/Homepage-PromptEnhancer-1abc9c?logo=homeassistant&logoColor=white" alt="Homepage"></a>
  <a href="https://github.com/Tencent-Hunyuan/HunyuanImage-2.1"><img src="https://img.shields.io/badge/Code-HunyuanImage2.1-2ecc71?logo=github" alt="HunyuanImage2.1 Code"></a>
  <a href="https://huggingface.co/tencent/HunyuanImage-2.1"><img src="https://img.shields.io/badge/Model-HunyuanImage2.1-3498db?logo=huggingface" alt="HunyuanImage2.1 Model"></a>
  <a href=https://x.com/TencentHunyuan target="_blank"><img src=https://img.shields.io/badge/Hunyuan-black.svg?logo=x height=22px></a>
</p>

---

<p align="center">
  <img src="assets/teaser-1.png" alt="PromptEnhancer Teaser"/>
</p>

## Overview

Hunyuan-PromptEnhancer is a prompt rewriting utility that **supports both Text-to-Image generation and Image-to-Image editing**. It restructures input prompts while preserving original intent, producing clearer, structured prompts for downstream image generation tasks.

**Key Features:**
- **Dual-mode support**: Text-to-Image prompt enhancement and Image-to-Image editing instruction refinement with visual context
- **Intent preservation**: Maintains all key elements (subject, action, style, layout, attributes, etc.) across rewriting
- **Robust parsing**: Multi-level fallback mechanism ensures reliable output
- **Flexible deployment**: Supports full-precision (7B/32B), quantized (GGUF), and vision-language models

## 🔥🔥🔥Updates

- [2025-09-30] ✨ Release [PromptEnhancer-Img2Img Editing model](https://huggingface.co/PromptEnhancer/PromptEnhancer-Img2img-Edit).
- [2025-09-22] 🚀 Thanks @mradermacher for adding **GGUF model support** for efficient inference with quantized models!
- [2025-09-18] ✨ Try the [PromptEnhancer-32B](https://huggingface.co/PromptEnhancer/PromptEnhancer-32B) for higher-quality prompt enhancement!
- [2025-09-16] ✨ Release [T2I-Keypoints-Eval dataset](https://huggingface.co/datasets/PromptEnhancer/T2I-Keypoints-Eval).
- [2025-09-07] ✨ Release [PromptEnhancer-7B model](https://huggingface.co/tencent/HunyuanImage-2.1/tree/main/reprompt).
- [2025-09-07] ✨ Release [technical report](https://arxiv.org/abs/2509.04545).

## Installation

### Option 1: Standard Installation (Recommended)
```bash
pip install -r requirements.txt
```

### Option 2: GGUF Installation (For quantized models with CUDA support)
```bash
chmod +x script/install_gguf.sh && ./script/install_gguf.sh
```

> **💡 Tip**: Choose GGUF installation if you want faster inference with lower memory usage, especially for the 32B model.

## Model Download

### 🎯 Quick Start

For most users, we recommend starting with the **PromptEnhancer-7B** model:

```bash
# Download PromptEnhancer-7B (13GB) - Best balance of quality and efficiency
huggingface-cli download tencent/HunyuanImage-2.1/reprompt --local-dir ./models/promptenhancer-7b
```

### 📊 Model Comparison & Selection Guide

| Model | Size | Quality | Memory | Best For |
|-------|------|---------|--------|----------|
| **PromptEnhancer-7B** | 13GB | High | 8GB+ | Most users, balanced performance |
| **PromptEnhancer-32B** | 64GB | Highest | 32GB+ | Research, highest quality needs |
| **32B-Q8_0 (GGUF)** | 35GB | Highest | 35GB+ | High-end GPUs (H100, A100) |
| **32B-Q6_K (GGUF)** | 27GB | Excellent | 27GB+ | RTX 4090, RTX 5090 |
| **32B-Q4_K_M (GGUF)** | 20GB | Good | 20GB+ | RTX 3090, RTX 4080 |

### Standard Models (Full Precision)
```bash
# PromptEnhancer-7B (recommended for most users)
huggingface-cli download tencent/HunyuanImage-2.1/reprompt --local-dir ./models/promptenhancer-7b

# PromptEnhancer-32B (for highest quality)
huggingface-cli download PromptEnhancer/PromptEnhancer-32B --local-dir ./models/promptenhancer-32b

# PromptEnhancer-Img2Img-Edit (for image editing tasks)
huggingface-cli download PromptEnhancer/PromptEnhancer-Img2img-Edit --local-dir ./models/promptenhancer-img2img-edit
```

### GGUF Models (Quantized - Memory Efficient)

Choose one based on your GPU memory:

```bash
# Q8_0: Highest quality (35GB)
huggingface-cli download mradermacher/PromptEnhancer-32B-GGUF PromptEnhancer-32B.Q8_0.gguf --local-dir ./models

# Q6_K: Excellent quality (27GB) - Recommended for RTX 4090
huggingface-cli download mradermacher/PromptEnhancer-32B-GGUF PromptEnhancer-32B.Q6_K.gguf --local-dir ./models

# Q4_K_M: Good quality (20GB) - Recommended for RTX 3090/4080
huggingface-cli download mradermacher/PromptEnhancer-32B-GGUF PromptEnhancer-32B.Q4_K_M.gguf --local-dir ./models
```

> **🚀 Performance Tip**: GGUF models offer 50-75% memory reduction with minimal quality loss. Use Q6_K for the best quality/memory trade-off.

## Quickstart

### Using HunyuanPromptEnhancer (Text-to-Image)

```python
from inference.prompt_enhancer import HunyuanPromptEnhancer

models_root_path = "./models/promptenhancer-7b"

enhancer = HunyuanPromptEnhancer(models_root_path=models_root_path, device_map="auto")

# Enhance a prompt (Chinese or English)
user_prompt = "Third-person view, a race car speeding on a city track..."
new_prompt = enhancer.predict(
    prompt_cot=user_prompt,
    # Default system prompt is tailored for image prompt rewriting; override if needed
    temperature=0.7,   # >0 enables sampling; 0 uses deterministic generation
    top_p=0.9,
    max_new_tokens=256,
)

print("Enhanced:", new_prompt)
```

### Using PromptEnhancerImg2Img (Image Editing)

For image editing tasks where you want to enhance editing instructions based on input images:

```python
from inference.prompt_enhancer_img2img import PromptEnhancerImg2Img

# Initialize the image-to-image prompt enhancer
enhancer = PromptEnhancerImg2Img(
    model_path="./models/your-model",
    device_map="auto"
)

# Enhance an editing instruction with image context
edit_instruction = "Remove the watermark from the bottom"
image_path = "./examples/sample_image.png"

enhanced_prompt = enhancer.predict(
    edit_instruction=edit_instruction,
    image_path=image_path,
    temperature=0.1,
    top_p=0.9,
    max_new_tokens=2048
)

print("Enhanced editing prompt:", enhanced_prompt)
```

### Using GGUF Models (Quantized, Faster)

```python
from inference.prompt_enhancer_gguf import PromptEnhancerGGUF

# Auto-detects Q8_0 model in models/ folder
enhancer = PromptEnhancerGGUF(
    model_path="./models/PromptEnhancer-32B.Q8_0.gguf",  # Optional: auto-detected
    n_ctx=1024,        # Context window size
    n_gpu_layers=-1,   # Use all GPU layers
)

# Enhance a prompt
user_prompt = "woman in jungle"
enhanced_prompt = enhancer.predict(
    user_prompt,
    temperature=0.3,
    top_p=0.9,
    max_new_tokens=512,
)

print("Enhanced:", enhanced_prompt)
```

### Command Line Usage (GGUF)

```bash
# Simple usage - auto-detects model in models/ folder
python inference/prompt_enhancer_gguf.py

# Or specify model path
GGUF_MODEL_PATH="./models/PromptEnhancer-32B.Q8_0.gguf" python inference/prompt_enhancer_gguf.py
```

## GGUF Model Benefits

🚀 **Why use GGUF models?**
- **Memory Efficient**: 50-75% less VRAM usage compared to full precision models
- **Faster Inference**: Optimized for CPU and GPU acceleration with llama.cpp
- **Quality Preserved**: Q8_0 and Q6_K maintain excellent output quality
- **Easy Deployment**: Single file format, no complex dependencies
- **GPU Acceleration**: Full CUDA support for high-performance inference

| Model | Size | Quality | VRAM Usage | Best For |
|-------|------|---------|------------|----------|
| Q8_0  | 35GB | Highest | ~35GB      | High-end GPUs (H100, A100) |
| Q6_K  | 27GB | Excellent | ~27GB     | RTX 4090, RTX 5090 |
| Q4_K_M| 20GB | Good    | ~20GB      | RTX 3090, RTX 4080 |

## Usage Comparison

| Model | Input Type | Use Case | Model Backend |
|-------|------------|----------|---------------|
| **HunyuanPromptEnhancer** | Text only | Text-to-Image generation | Transformers (7B/32B) |
| **PromptEnhancerImg2Img** | Text + Image | Image editing tasks | Transformers (32B) |
| **PromptEnhancerGGUF** | Text only | Memory-efficient T2I | llama.cpp (quantized) |

## Parameters

### Standard Models (Transformers)
- `models_root_path`: Local path or repo id; supports `trust_remote_code` models.
- `device_map`: Device mapping (default `auto`).
- `predict(...)`:
  - `prompt_cot` (str): Input prompt to rewrite.
  - `sys_prompt` (str): Optional system prompt; a default is provided for image prompt rewriting.
  - `temperature` (float): `>0` enables sampling; `0` for deterministic generation.
  - `top_p` (float): Nucleus sampling threshold (effective when sampling).
  - `max_new_tokens` (int): Maximum number of new tokens to generate.

### GGUF Models
- `model_path` (str): Path to GGUF model file (auto-detected if in models/ folder).
- `n_ctx` (int): Context window size (default: 8192, recommended: 1024 for short prompts).
- `n_gpu_layers` (int): Number of layers to offload to GPU (-1 for all layers).
- `verbose` (bool): Enable verbose logging from llama.cpp.

### Image-to-Image Models (PromptEnhancerImg2Img)
- `model_path` (str): Path to the pretrained Qwen2.5-VL model.
- `device_map` (str): Device mapping for model loading (default: `auto`).
- `predict(...)`:
  - `edit_instruction` (str): Original editing instruction.
  - `image_path` (str): Path to the input image file.
  - `sys_prompt` (str): Optional system prompt (uses default if None).
  - `temperature` (float): Sampling temperature (default: 0.1).
  - `top_p` (float): Nucleus sampling threshold (default: 0.9).
  - `max_new_tokens` (int): Maximum tokens to generate (default: 2048).

## Citation

If you find this project useful, please consider citing:
```bibtex
@article{promptenhancer,
  title={PromptEnhancer: A Simple Approach to Enhance Text-to-Image Models via Chain-of-Thought Prompt Rewriting},
  author={Wang, Linqing and Xing, Ximing and Cheng, Yiji and Zhao, Zhiyuan and Tao, Jiale and Wang, QiXun and Li, Ruihuang and Chen, Comi and Li, Xin and Wu, Mingrui and Deng, Xinchi and Wang, Chunyu and Lu, Qinglin},
  journal={arXiv preprint arXiv:2509.04545},
  year={2025}
}
```

## Acknowledgements

We would like to thank the following open-source projects and communities for their contributions to open research and exploration: [Transformers](https://huggingface.co/transformers) and [HuggingFace](https://huggingface.co).

## Contact

If you would like to leave a message for our R&D and product teams, Welcome to contact our open-source team. You can also contact us via email (hunyuan_opensource@tencent.com).

## Github Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Hunyuan-PromptEnhancer/PromptEnhancer&type=Date)](https://www.star-history.com/#Hunyuan-PromptEnhancer/PromptEnhancer&Date)