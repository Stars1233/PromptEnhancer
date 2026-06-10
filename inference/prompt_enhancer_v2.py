"""
Copyright (c) 2025 Tencent. All Rights Reserved.
Licensed under the Tencent Hunyuan Community License Agreement.
"""

import re
import os
import time
import logging

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def replace_single_quotes(text):
    """
    Replace single quotes within words with double quotes, and convert
    curly single quotes to curly double quotes for consistency.
    """
    pattern = r"\B'([^']*)'\B"
    replaced_text = re.sub(pattern, r'"\1"', text)
    replaced_text = replaced_text.replace("’", "”")
    replaced_text = replaced_text.replace("‘", "“")
    return replaced_text

class PromptEnhancerV2:

    def __init__(self, models_root_path, device_map="auto"):
        """
        Initialize the PromptEnhancerV2 class with model and processor.

        Args:
            models_root_path (str): Path to the pretrained model.
            device_map (str): Device mapping for model loading.
        """
        # Lazy logging setup (will be no-op if already configured by app)
        if not logging.getLogger(__name__).handlers:
            logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            models_root_path,
            torch_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2",
            device_map=device_map,
        )
        self.processor = AutoProcessor.from_pretrained(models_root_path)

    @torch.inference_mode()
    def predict(
        self,
        prompt_cot,
        sys_prompt="请根据用户的输入，生成思考过程的思维链并改写提示词：",
        temperature=0,
        top_p=1.0,
        max_new_tokens=2048,
        device="cuda",
    ):
        """
        Generate a rewritten prompt using the model.

        Args:
            prompt_cot (str): The original prompt to be rewritten.
            sys_prompt (str): System prompt to guide the rewriting.
            temperature (float): Sampling temperature.
            top_p (float): Top-p sampling parameter.
            max_new_tokens (int): Maximum number of new tokens to generate.
            device (str): Device for inference.

        Returns:
            str: The rewritten prompt, or the original if generation fails.
        """
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
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=2048,
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
        except Exception:
            prompt_cot = org_prompt_cot
            print("✗ Re-prompting failed, so we are using the original prompt")

        return prompt_cot

if __name__ == "__main__":
    model_path = os.environ.get('MODEL_OUTPUT_PATH', "/apdcephfs_zwfy3/share_302243908/aladdinwang/001_weight/models/promptenhancer-32b")

    prompt_enhancer_cls = PromptEnhancerV2(
        models_root_path=model_path
    )

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

    print("Testing Chinese prompts:")
    for item in test_list_zh:
        print("User Prompt:", item)
        print("---------:")
        time_start = time.time()
        result = prompt_enhancer_cls.predict(item)
        time_end = time.time()
        print("RePrompt:", result)
        print("Time cost:", time_end - time_start)
        print("~~~~~~~~~~~~~~")

    print("\nTesting English prompts:")
    for item in test_list_en:
        print("User Prompt:", item)
        print("---------:")
        time_start = time.time()
        result = prompt_enhancer_cls.predict(item)
        time_end = time.time()
        print("RePrompt:", result)
        print("Time cost:", time_end - time_start)
        print("~~~~~~~~~~~~~~")