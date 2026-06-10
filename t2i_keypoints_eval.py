# -*- coding: utf-8 -*-
"""Evaluate a prompt enhancer on the T2I-Keypoints-Eval benchmark.

The benchmark measures how well a prompt enhancer improves text-to-image (T2I)
generation. Every sample carries a raw prompt together with a list of
fine-grained "key points" (and their descriptions) that a faithful image should
satisfy. A vision-language model (Gemini by default) acts as the judge: for each
key point it inspects the generated image and decides ``accept`` or ``reject``.

Pipeline per sample:
    1. Enhance the raw prompt              -> ``prompt_enhancer``
    2. Generate an image from the prompt   -> ``text_to_image``
    3. Ask the judge whether the image     -> ``GeminiJudge``
       satisfies each key point.
    4. Aggregate per-key-point and overall accuracy.

To benchmark your own system, implement ``prompt_enhancer`` and
``text_to_image`` below. The default implementations are no-op placeholders.

Setup:
    pip install -r requirements.txt
    export GEMINI_API_KEY="your-key"   # https://aistudio.google.com/apikey
    python eval_pe.py --num-samples 10
"""

import argparse
import json
import logging
import os
import time
from mimetypes import guess_type
from typing import Optional, Union

import requests
from datasets import load_dataset
from tqdm import tqdm

try:
    from google import genai
    from google.genai import types
except ImportError:  # pragma: no cover - surfaced lazily when the judge is used
    genai = None
    types = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("eval_pe")

DATASET_NAME = "PromptEnhancer/T2I-Keypoints-Eval"


def write_json(data, path: str, encoding: str = "utf-8") -> None:
    with open(path, "w", encoding=encoding) as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# --------------------------------------------------------------------------- #
# Vision-language judge (Google Gemini official API)
# --------------------------------------------------------------------------- #
def _load_image_bytes(image: str) -> tuple[bytes, str]:
    """Return ``(raw_bytes, mime_type)`` for a remote URL or a local file path."""
    if image.startswith(("http://", "https://")):
        resp = requests.get(image, timeout=60)
        resp.raise_for_status()
        mime_type = resp.headers.get("Content-Type") or guess_type(image)[0]
        return resp.content, mime_type or "image/jpeg"

    with open(image, "rb") as f:
        data = f.read()
    return data, guess_type(image)[0] or "image/jpeg"


class GeminiJudge:
    """Calls the Google Gemini API to judge whether an image meets a key point."""

    def __init__(
        self,
        model: str = "gemini-2.5-pro",
        api_key: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        if genai is None:
            raise ImportError(
                "google-genai is required for the default judge. "
                "Install it with `pip install google-genai`."
            )
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "No Gemini API key found. Set the GEMINI_API_KEY environment "
                "variable or pass api_key. Get a key at "
                "https://aistudio.google.com/apikey"
            )
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def __call__(self, instruction: str, image: str) -> Optional[str]:
        """Return the judge's raw text response, or ``None`` if all retries fail."""
        data, mime_type = _load_image_bytes(image)
        image_part = types.Part.from_bytes(data=data, mime_type=mime_type)

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=[image_part, instruction],
                )
                if response.text:
                    return response.text
            except Exception as exc:  # noqa: BLE001 - retry on any API error
                logger.warning("Judge call failed (attempt %d): %s", attempt, exc)
            time.sleep(self.retry_delay)
        return None


def parse_verdict(message: Optional[str]) -> int:
    """Map a judge response to 1 (accept) or 0 (reject).

    The judge is asked to reason first and state its verdict last, so we use the
    last occurrence of "accept"/"reject" rather than a naive substring check.
    """
    if not message:
        return 0
    text = message.lower()
    accept_at = text.rfind("accept")
    reject_at = text.rfind("reject")
    if accept_at == -1 and reject_at == -1:
        return 0
    return 1 if accept_at > reject_at else 0


# --------------------------------------------------------------------------- #
# Plug in your own prompt enhancer / T2I model here
# --------------------------------------------------------------------------- #
def prompt_enhancer(prompt: str) -> str:
    """Enhance a raw prompt. Replace with your own enhancer."""
    return prompt


def text_to_image(prompt: str) -> str:
    """Generate an image for ``prompt`` and return a URL or local file path.

    Replace this with a call to your text-to-image model. The returned value is
    passed directly to the judge, so it must be a reachable URL or a local path.
    """
    raise NotImplementedError(
        "Implement text_to_image() to generate an image from the (enhanced) "
        "prompt and return its URL or local file path."
    )


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def build_instruction(prompt: str, key_point: str, key_point_desc: str, language: str) -> str:
    """Build the judge instruction for a single key point."""
    if language == "zh":
        return (
            f"该图片由 prompt:[{prompt}] 通过 AIGC 生成。请判断，基于图片画面中的可见部分，"
            f"只需要考虑 [{key_point}] 是否严格满足 [{key_point_desc}] 要求。"
            f"accept 代表满足要求，reject 代表不满足要求。"
            f"仅根据图片视觉呈现的可见部分（不要联想画面，不要猜测画面），"
            f"详细分析得到结果的依据并给出结果："
        )
    return (
        f"This image is generated by prompt:[{prompt}] through AIGC. Please judge, "
        f"based on the visible part of the image, only consider whether [{key_point}] "
        f"strictly meets the [{key_point_desc}] requirement. accept means meeting the "
        f"requirement, reject means not meeting the requirement. Only rely on the "
        f"visible part of the image (do not associate the scene & do not guess the "
        f"scene), analyze the basis for the result in detail and give the result:"
    )


def eval_sample(sample: dict, judge: GeminiJudge) -> dict:
    """Run the full pipeline for one benchmark sample and attach the scores."""
    prompt = sample["prompt"]
    enhanced_prompt = prompt_enhancer(prompt)
    sample["reprompt"] = enhanced_prompt

    image = text_to_image(enhanced_prompt)

    scores = []
    for key_point, key_point_desc in zip(
        sample["prompt_points"], sample["prompt_points_des"]
    ):
        instruction = build_instruction(
            prompt, key_point, key_point_desc, sample["language"]
        )
        verdict = judge(instruction, image)
        scores.append(parse_verdict(verdict))

    sample["res_score_list"] = scores
    return sample


def report(results: list) -> dict:
    """Print and return per-key-point accuracy plus the all-points-correct ratio."""
    benchmark: dict = {}
    all_correct = 0

    for item in results:
        scores = item["res_score_list"]
        if scores and 0 not in scores:
            all_correct += 1
        for key_point, score in zip(item["prompt_points"], scores):
            stats = benchmark.setdefault(key_point, {"all": 0, "right": 0})
            stats["all"] += 1
            stats["right"] += score

    total = len(results) or 1
    logger.info(
        "All key points correct: %d / %d (%.4f)",
        all_correct,
        len(results),
        all_correct / total,
    )
    for key_point, stats in benchmark.items():
        logger.info("%s: %.4f", key_point, stats["right"] / stats["all"])

    return {
        "all_correct": all_correct,
        "total": len(results),
        "all_correct_ratio": all_correct / total,
        "per_key_point": {
            k: v["right"] / v["all"] for k, v in benchmark.items()
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="gemini-2.5-pro",
        help="Gemini model used as the judge (default: gemini-2.5-pro).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Gemini API key. Defaults to the GEMINI_API_KEY environment variable.",
    )
    parser.add_argument(
        "--split", default="train", help="Dataset split to evaluate (default: train)."
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Limit the number of samples to evaluate (default: all).",
    )
    parser.add_argument(
        "--output",
        default="all_res_eval.json",
        help="Where to save per-sample results (default: all_res_eval.json).",
    )
    parser.add_argument(
        "--max-retries", type=int, default=3, help="Max judge retries per call."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    judge = GeminiJudge(
        model=args.model, api_key=args.api_key, max_retries=args.max_retries
    )

    dataset = load_dataset(DATASET_NAME)[args.split]
    if args.num_samples is not None:
        dataset = dataset.select(range(min(args.num_samples, len(dataset))))

    results = [eval_sample(sample, judge) for sample in tqdm(dataset, desc="Evaluating")]

    write_json(results, args.output)
    logger.info("Saved per-sample results to %s", args.output)
    report(results)


if __name__ == "__main__":
    main()
