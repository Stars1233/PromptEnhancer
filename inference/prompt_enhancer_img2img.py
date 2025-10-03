"""
Copyright (c) 2025 Tencent. All Rights Reserved.
Licensed under the Tencent Hunyuan Community License Agreement.

Image-to-Image Prompt Enhancer for image editing tasks.
"""

import time
import logging

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


class PromptEnhancerImg2Img:
    """
    Prompt enhancer for image editing tasks using Qwen2.5-VL model.

    This class enhances editing instructions by analyzing the input image
    and generating detailed prompts that preserve non-edited regions.
    """

    # Image resolution constraints (in pixels)
    MIN_PIXELS = 512 * 28 * 28
    MAX_PIXELS = 1280 * 28 * 28

    # Default system prompt for image editing
    # DEFAULT_SYS_PROMPT = (
    #     "请根据用户的输入图片和编辑指令,参考图片对编辑指令进行改写,"
    #     "要求非编辑区域保持不变。"
    # )
    DEFAULT_SYS_PROMPT = '请根据用户的输入图片和编辑指令，参考图片对编辑指令进行改写，要求非编辑区域保持不变。'

    def __init__(
        self,
        model_path: str,
        device_map: str = "auto"
    ):
        """
        Initialize the Image-to-Image Prompt Enhancer.

        Args:
            model_path (str): Path to the pretrained Qwen2.5-VL model.
            device_map (str): Device mapping for model loading (default: "auto").
        """
        # Lazy logging setup
        if not logging.getLogger(__name__).handlers:
            logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.model_path = model_path
        self.logger.info(f"Loading model from: {model_path}")

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=device_map
        )

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.logger.info("Model loaded successfully")

    @torch.inference_mode()
    def predict(
        self,
        edit_instruction: str,
        image_path: str,
        sys_prompt: str = None,
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_new_tokens: int = 2048,
    ) -> str:
        """
        Generate enhanced editing prompt based on input image and instruction.

        Args:
            edit_instruction (str): Original editing instruction.
            image_path (str): Path to the input image.
            sys_prompt (str): Optional system prompt (uses default if None).
            temperature (float): Sampling temperature (default: 0.1).
            top_p (float): Top-p sampling parameter (default: 0.9).
            max_new_tokens (int): Maximum tokens to generate (default: 2048).

        Returns:
            str: Enhanced editing prompt with detailed instructions.
        """
        # Use default system prompt if not provided
        if sys_prompt is None:
            sys_prompt = self.DEFAULT_SYS_PROMPT

        # Format user prompt with image and instruction
        user_prompt = f"输入图片为:<image>,编辑指令为:{edit_instruction}"

        messages = [
            {"role": "system", "content": sys_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image", "image": image_path},
                ],
            },
        ]

        # Prepare inputs for the model
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # Generate enhanced prompt
        time_start = time.time()
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=top_p
        )
        time_elapsed = time.time() - time_start

        self.logger.info(f"Inference time: {time_elapsed:.2f}s")

        # Decode generated tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        return output_text[0]


if __name__ == "__main__":
    # Example usage
    model_path = "/path/to/your/qwen-vl-model"

    enhancer = PromptEnhancerImg2Img(
        model_path=model_path,
        device_map="auto"
    )

    # Test cases for image editing
    test_cases = [
        {
            "instruction": "把食物前面白色的标签去掉",
            "image": "./examples/image1.png"
        },
        {
            "instruction": "去除图片中的水印",
            "image": "./examples/image2.png"
        },
        {
            "instruction": "把黑板上的字改成'强化学习',再把图片下方的水印去掉",
            "image": "./examples/image3.png"
        },
    ]

    for idx, case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test Case {idx}/{len(test_cases)}")
        print(f"{'='*60}")
        print(f"Instruction: {case['instruction']}")
        print(f"Image: {case['image']}")
        print(f"{'-'*60}")

        enhanced_prompt = enhancer.predict(
            edit_instruction=case['instruction'],
            image_path=case['image']
        )

        print(f"Enhanced Prompt:\n{enhanced_prompt}")
        print(f"{'='*60}\n")