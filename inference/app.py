# app.py
# Gradio UI for PromptEnhancerV2

import os
import time
import logging
import re
import torch
import gradio as gr

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# 尝试导入 qwen_vl_utils，若失败则提供降级实现（返回空的图像/视频输入）
try:
    from qwen_vl_utils import process_vision_info
except Exception:
    def process_vision_info(messages):
        return None, None

def replace_single_quotes(text):
    pattern = r"\B'([^']*)'\B"
    replaced_text = re.sub(pattern, r'"\1"', text)
    replaced_text = replaced_text.replace("’", "”").replace("‘", "“")
    return replaced_text

class PromptEnhancerV2:
    def __init__(self, models_root_path, device_map="auto", torch_dtype="bfloat16"):
        if not logging.getLogger(__name__).handlers:
            logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # dtype 兼容处理
        if torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        elif torch_dtype == "float16":
            dtype = torch.float16
        else:
            dtype = torch.float32

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            models_root_path,
            torch_dtype=dtype,
            attn_implementation="flash_attention_2",
            device_map=device_map,
        )
        self.processor = AutoProcessor.from_pretrained(models_root_path)

    @torch.inference_mode()
    def predict(
        self,
        prompt_cot,
        sys_prompt="请根据用户的输入，生成思考过程的思维链并改写提示词：",
        temperature=0.0,
        top_p=1.0,
        max_new_tokens=2048,
        device="cuda",
    ):
        org_prompt_cot = prompt_cot
        try:
            user_prompt_format = sys_prompt + "\n" + org_prompt_cot
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt_format},
                    ],
                }
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(device)

            # 注意：原始代码固定 do_sample=False，top_k=5, top_p=0.9，这里保持一致
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=2048,  # 与原始代码保持一致（未使用 max_new_tokens 参数）
                temperature=float(temperature),
                do_sample=False,
                top_k=5,
                top_p=0.9
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            output_res = output_text[0]
            assert output_res.count("think>") == 2
            prompt_cot = output_res.split("think>")[-1]
            if prompt_cot.startswith("\n"):
                prompt_cot = prompt_cot[1:]
            prompt_cot = replace_single_quotes(prompt_cot)
        except Exception as e:
            prompt_cot = org_prompt_cot
            print(f"✗ Re-prompting failed, so we are using the original prompt. Error: {e}")

        return prompt_cot


# -------------------------
# Gradio app helpers
# -------------------------

DEFAULT_MODEL_PATH = os.environ.get("MODEL_OUTPUT_PATH", "/path/to/your/qwen-model")

def ensure_enhancer(state, model_path, device_map, torch_dtype):
    """
    state: dict or None
    Returns: (state_dict)
    """
    need_reload = False
    if state is None or not isinstance(state, dict):
        need_reload = True
    else:
        prev_path = state.get("model_path")
        prev_map = state.get("device_map")
        prev_dtype = state.get("torch_dtype")
        if prev_path != model_path or prev_map != device_map or prev_dtype != torch_dtype:
            need_reload = True

    if need_reload:
        enhancer = PromptEnhancerV2(model_path, device_map=device_map, torch_dtype=torch_dtype)
        return {"enhancer": enhancer, "model_path": model_path, "device_map": device_map, "torch_dtype": torch_dtype}
    return state

def run_single(prompt, sys_prompt, temperature, max_new_tokens, device,
               model_path, device_map, torch_dtype, state):
    if not prompt or not str(prompt).strip():
        return "", "请先输入提示词。", state

    t0 = time.time()
    state = ensure_enhancer(state, model_path, device_map, torch_dtype)
    enhancer = state["enhancer"]
    try:
        out = enhancer.predict(
            prompt_cot=prompt,
            sys_prompt=sys_prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            device=device
        )
        dt = time.time() - t0
        return out, f"耗时：{dt:.2f}s", state
    except Exception as e:
        return "", f"推理失败：{e}", state

def run_batch(batch_text, sys_prompt, temperature, max_new_tokens, device,
              model_path, device_map, torch_dtype, state):
    """
    batch_text: 多行文本，每行一个提示词
    """
    if not batch_text or not str(batch_text).strip():
        return "", "请在左侧输入区每行填写一个提示词。", state

    lines = [l.strip() for l in batch_text.splitlines() if l.strip()]
    if not lines:
        return "", "未检测到有效提示词。", state

    state = ensure_enhancer(state, model_path, device_map, torch_dtype)
    enhancer = state["enhancer"]

    results = []
    total_t0 = time.time()
    for i, line in enumerate(lines, 1):
        t0 = time.time()
        try:
            out = enhancer.predict(
                prompt_cot=line,
                sys_prompt=sys_prompt,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                device=device
            )
            dt = time.time() - t0
            results.append(f"[{i}] 原始: {line}\n    重写: {out}\n    耗时: {dt:.2f}s\n")
        except Exception as e:
            results.append(f"[{i}] 原始: {line}\n    失败: {e}\n")
    total_dt = time.time() - total_t0
    summary = f"共处理 {len(lines)} 条，累计耗时：{total_dt:.2f}s"
    return "\n".join(results), summary, state


# 示例数据
test_list_zh = [
    "一幅书法作品，上边写着'生于忧患，死于安乐。'",
    "第三人称视角，赛车在城市赛道上飞驰，左上角是小地图，地图下面是当前名次，右下角仪表盘显示当前速度。",
    "韩系插画风女生头像，粉紫色短发+透明感腮红，侧光渲染。",
    "点彩派，盛夏海滨，两位渔夫正在搬运木箱，三艘帆船停在岸边，对角线构图。",
    "一幅由梵高绘制的梦境麦田，旋转的蓝色星云与燃烧的向日葵相纠缠。",
]
test_list_en = [
    "Create a painting depicting a 30-year-old white female white-collar worker on a business trip by plane.",
    "Depicted in the anime style of Studio Ghibli, a girl stands quietly at the deck with a gentle smile.",
    "Blue background, a lone girl gazes into the distant sea; her expression is sorrowful.",
    "A blend of expressionist and vintage styles, drawing a building with colorful walls.",
    "Paint a winter scene with crystalline ice hangings from an Antarctic research station.",
]

with gr.Blocks(title="Prompt Enhancer_V2") as demo:
    gr.Markdown("## 提示词重写器")
    with gr.Row():
        with gr.Column(scale=2):
            model_path = gr.Textbox(
                label="模型路径（本地或HF地址）",
                value=DEFAULT_MODEL_PATH,
                placeholder="/apdcephfs_jn3/share_302243908/aladdinwang/model_weight/cot_taurus_v6_50/global_step0",
            )
            device_map = gr.Dropdown(
                choices=["auto", "cuda", "cpu"],
                value="auto",
                label="device_map（模型加载映射）"
            )
            torch_dtype = gr.Dropdown(
                choices=["bfloat16", "float16", "float32"],
                value="bfloat16",
                label="torch_dtype"
            )

        with gr.Column(scale=3):
            sys_prompt = gr.Textbox(
                label="系统提示词",
                value="请根据用户的输入，生成思考过程的思维链并改写提示词：",
                lines=3
            )
            with gr.Row():
                temperature = gr.Slider(0, 1, value=0.0, step=0.05, label="Temperature（原代码 do_sample=False 时不生效）")
                max_new_tokens = gr.Slider(16, 4096, value=2048, step=16, label="Max New Tokens（原代码未使用该参数）")
                device = gr.Dropdown(choices=["cuda", "cpu"], value="cuda", label="推理device")

    state = gr.State(value=None)

    with gr.Tab("单条推理"):
        with gr.Row():
            with gr.Column(scale=2):
                prompt = gr.Textbox(label="输入提示词", lines=6, placeholder="在此粘贴要改写的提示词...")
                run_btn = gr.Button("生成重写", variant="primary")
                gr.Examples(
                    examples=test_list_zh + test_list_en,
                    inputs=prompt,
                    label="示例"
                )
            with gr.Column(scale=3):
                out_text = gr.Textbox(label="重写结果", lines=10)
                out_info = gr.Markdown("准备就绪。")

        run_btn.click(
            run_single,
            inputs=[prompt, sys_prompt, temperature, max_new_tokens, device,
                    model_path, device_map, torch_dtype, state],
            outputs=[out_text, out_info, state]
        )

    gr.Markdown(
        "提示：若需使温度等采样参数生效，请在 PromptEnhancerV2.generate 内将 do_sample 设置为 True。"
    )

# 为避免多并发导致显存爆，限制并发
# demo.queue(concurrency_count=1, max_size=10)
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080, show_error=True)