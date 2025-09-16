#!/usr/bin/env python3
"""
Run ALL models concurrently:
- Exact GenAI-Bench prompt loading (GenAIBench_Image) & indexing
- API-based image generation (imagen4, gemini2.0, dall-e3, sd-* ports)
- GPT-4o VQAScore via t2v_metrics (CPU; no GPU required)
- Per-model worker runs generation + scoring, in parallel across models

Outputs per model:
  ./outputs/<model>/<prompt_idx>.png
  ./results_vqa/<model>_gpt4o_vqa_per_prompt.json
  ./results_vqa/<model>_gpt4o_vqa_summary.json
And a combined:
  ./results_vqa/leaderboard_gpt4o_vqa.json
"""

import argparse
import base64
import io
import json
import os
import pathlib
import re
import time
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

import t2v_metrics
from t2v_metrics.dataset import GenAIBench_Image  # your local path/module

# --------------------- ENV VARS ---------------------
OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY", "")

# yunwu OpenAI-compatible (for DALL·E-3)
YUNWU_BASE_URL       = os.getenv("YUNWU_BASE_URL", "https://yunwu.ai/v1")
YUNWU_API_KEY        = os.getenv("YUNWU_API_KEY", OPENAI_API_KEY)

# Gemini 2.0 (yunwu generateContent)
GEMINI_URL           = os.getenv("GEMINI_URL", "https://yunwu.ai/v1beta/models/gemini-2.0-flash-preview-image-generation:generateContent")
GEMINI_API_KEY       = os.getenv("GEMINI_API_KEY", "")
GEMINI_25_URL = os.getenv("GEMINI_25_URL", "https://yunwu.ai/v1beta/models/gemini-2.5-flash-image-preview:generateContent")

# Imagen-4 (yunwu/replicate)
IMAGEN_SUBMIT_URL    = os.getenv("IMAGEN_SUBMIT_URL", "https://yunwu.ai/replicate/v1/models/google/imagen-4/predictions")
IMAGEN_POLL_BASE     = os.getenv("IMAGEN_POLL_BASE",  "https://yunwu.ai/replicate/v1/predictions")
IMAGEN_API_KEY       = os.getenv("IMAGEN_API_KEY", "")

# Stable Diffusion server
SD_HOST              = os.getenv("SD_HOST", "101.6.69.18")
SD_SCHEME            = os.getenv("SD_SCHEME", "http")
SD_ENDPOINT_SUFFIX   = os.getenv("SD_ENDPOINT_SUFFIX", "generate/")  # e.g., '/generate/'
SD_TIMEOUT           = float(os.getenv("SD_TIMEOUT", "180"))
SD_BASE_09_PORT      = os.getenv("SD_BASE_09_PORT", "10001")
SD_BASE_10_PORT      = os.getenv("SD_BASE_10_PORT", "10000")
SD35_MEDIUM_PORT     = os.getenv("SD35_MEDIUM_PORT", "10003")
SD35_LARGE_PORT      = os.getenv("SD35_LARGE_PORT", "10002")

# --------------------- UTILS ---------------------

SAFE_IMG_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}

def ensure_dir(p: pathlib.Path):
    p.mkdir(parents=True, exist_ok=True)

def clean_filename(text: str, max_length: int = 80) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*]', "_", text)
    cleaned = re.sub(r'[，。！？、\s]+', "_", cleaned)
    cleaned = cleaned[:max_length].strip("_")
    return cleaned or "image"

def save_image_bytes(img_bytes: bytes, out_path: pathlib.Path) -> pathlib.Path:
    # Robust image decode with PIL; force PNG output (t2v_metrics accepts it)
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except UnidentifiedImageError as e:
        raise RuntimeError(f"Unidentified image bytes: {e}")
    out_path = out_path.with_suffix(".png")
    img.save(out_path, format="PNG", optimize=True)
    return out_path

def is_under_15mb(path: pathlib.Path) -> bool:
    try:
        return path.stat().st_size < 15 * 1024 * 1024
    except FileNotFoundError:
        return False

def basic_prompt_guard(prompt: str) -> bool:
    # Optional: keep you out of obvious policy trouble with vendors. Adjust/remove if not needed.
    lowered = prompt.lower()
    blocked = ["sexual", "porn", "nsfw", "nude", "child sexual", "csam", "self-harm", "suicide", "bomb", "explosive", "bioweapon"]
    return not any(k in lowered for k in blocked)

def index_to_stem(idx, width: int = 5) -> str:
    """Return zero-padded stem for filenames from a GenAI-Bench index (int or str)."""
    try:
        return f"{int(idx):0{width}d}"
    except Exception:
        s = str(idx)
        return s if len(s) >= width else s.zfill(width)

def to_int_or_keep(x):
    try:
        return int(x)
    except Exception:
        return x

def load_genai_bench_prompts(root_dir: str, num_prompts: int) -> List[Tuple[str, str]]:
    """
    EXACTLY like official code: GenAIBench_Image index -> prompt text.
    Return list of (original_key, prompt). Keys can be str or int; we keep them as-is.
    """
    ds = GenAIBench_Image(root_dir=root_dir, num_prompts=num_prompts)
    # Prefer numeric sort if keys are numeric-like
    try:
        idxs = sorted(ds.dataset.keys(), key=lambda k: int(k))
    except Exception:
        idxs = sorted(ds.dataset.keys())
    return [(i, ds.dataset[i]['prompt']) for i in idxs]

# --------------------- GENERATORS (API adapters) ---------------------

def gen_gemini20_image(prompt: str) -> bytes:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set for Gemini 2.0 generation.")
    url = f"{GEMINI_URL}?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
    }
    r = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(payload), timeout=180)
    r.raise_for_status()
    resp = r.json()
    for cand in resp.get("candidates", []):
        for part in cand.get("content", {}).get("parts", []):
            if "inlineData" in part and "data" in part["inlineData"]:
                return base64.b64decode(part["inlineData"]["data"])
    raise RuntimeError("Gemini response did not include inline image bytes.")

# gemini-2.5-flash-image-preview
def gen_gemini25_image(prompt: str) -> bytes:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set for Gemini 2.0 generation.")
    url = f"{GEMINI_25_URL}?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
    }
    r = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(payload), timeout=180)
    r.raise_for_status()
    resp = r.json()
    for cand in resp.get("candidates", []):
        for part in cand.get("content", {}).get("parts", []):
            if "inlineData" in part and "data" in part["inlineData"]:
                return base64.b64decode(part["inlineData"]["data"])
    raise RuntimeError("Gemini response did not include inline image bytes.")

def gen_dalle3_yunwu(prompt: str, size: str = "1024x1024") -> bytes:
    if not YUNWU_API_KEY:
        raise RuntimeError("YUNWU_API_KEY (or OPENAI_API_KEY) required for yunwu OpenAI-compatible generation.")
    url = f"{YUNWU_BASE_URL.rstrip('/')}/images/generations"
    headers = {"Authorization": f"Bearer {YUNWU_API_KEY}"}
    payload = {"model": "dall-e-3", "prompt": prompt, "size": size, "n": 1, "response_format": "b64_json"}
    r = requests.post(url, headers=headers, json=payload, timeout=180)
    r.raise_for_status()
    js = r.json()
    if js.get("data"):
        item = js["data"][0]
        if "b64_json" in item:
            return base64.b64decode(item["b64_json"])
        if "url" in item:
            img_r = requests.get(item["url"], timeout=180)
            img_r.raise_for_status()
            return img_r.content
    raise RuntimeError("Unexpected DALL·E response shape from yunwu.")

def gen_gpt4o_yunwu(prompt: str, size: str = "1024x1024", model: str = "gpt-4o-image-vip") -> bytes:
    if not YUNWU_API_KEY:
        raise RuntimeError("YUNWU_API_KEY (or OPENAI_API_KEY) required for Yunwu image generation.")

    url = f"{YUNWU_BASE_URL.rstrip('/')}/images/generations"
    headers = {"Authorization": f"Bearer {YUNWU_API_KEY}"}
    payload = {
        "model": model,
        "prompt": prompt,
        "size": size,
        "n": 1,
        "response_format": "b64_json",
        # Optional fields you can add if supported:
        "quality": "standard",
        # "background": "transparent",
        # "style": "vivid",
    }

    r = requests.post(url, headers=headers, json=payload, timeout=180)
    r.raise_for_status()
    js = r.json()

    # Standard OpenAI-compatible response: {"data": [{"b64_json": "..."}]}
    if isinstance(js, dict) and js.get("data"):
        item = js["data"][0]
        if "b64_json" in item:
            return base64.b64decode(item["b64_json"])
        if "url" in item:
            img_r = requests.get(item["url"], timeout=180)
            img_r.raise_for_status()
            return img_r.content
    raise RuntimeError("Unexpected GPT-4o image response shape from Yunwu.")

def gen_imagen4_yunwu(prompt: str, aspect_ratio: str = "1:1", output_format: str = "jpg") -> bytes:
    if not IMAGEN_API_KEY:
        raise RuntimeError("IMAGEN_API_KEY not set for Imagen-4 flow.")
    headers = {"Authorization": f"Bearer {IMAGEN_API_KEY}", "Content-Type": "application/json"}
    submit_payload = {"input": {"prompt": prompt, "aspect_ratio": aspect_ratio, "output_format": output_format, "safety_filter_level": "block_only_high"}}
    s = requests.post(IMAGEN_SUBMIT_URL, headers=headers, data=json.dumps(submit_payload), timeout=180)
    s.raise_for_status()
    task_id = s.json().get("id")
    if not task_id:
        raise RuntimeError("Imagen submit did not return a task id.")
    poll_url = f"{IMAGEN_POLL_BASE.rstrip('/')}/{task_id}"
    for _ in range(60):
        g = requests.get(poll_url, headers=headers, timeout=30)
        g.raise_for_status()
        js = g.json()
        status = js.get("status")
        if status == "succeeded":
            output = js.get("output")
            image_url = output[0] if isinstance(output, list) and output else output
            if not image_url:
                raise RuntimeError("Imagen succeeded but no output URL.")
            img_r = requests.get(image_url, timeout=180)
            img_r.raise_for_status()
            return img_r.content
        if status == "failed":
            raise RuntimeError(f"Imagen task failed: {js.get('error')}")
        time.sleep(5)
    raise RuntimeError("Imagen task timed out (5 minutes).")

def gen_sd_server(prompt: str, port: str, params: Optional[Dict] = None) -> bytes:
    if not port:
        raise RuntimeError("Missing SD port.")
    url = f"{SD_SCHEME}://{SD_HOST}:{port}/{SD_ENDPOINT_SUFFIX.lstrip('/')}"
    payload = {"prompt": prompt}
    if params:
        payload.update(params)
    r = requests.post(url, json=payload, timeout=SD_TIMEOUT)
    r.raise_for_status()
    ctype = r.headers.get("Content-Type", "")
    if "application/json" in ctype:
        js = r.json()
        if "image_base64" in js:
            return base64.b64decode(js["image_base64"])
        if "image_url" in js:
            img_url = js["image_url"]
            if img_url.startswith("/"):
                img_url = f"{SD_SCHEME}://{SD_HOST}:{port}{img_url}"
            img_r = requests.get(img_url, timeout=SD_TIMEOUT)
            img_r.raise_for_status()
            return img_r.content
        raise RuntimeError("SD JSON missing 'image_base64'/'image_url'.")
    return r.content  # raw bytes

MODEL_ADAPTERS = {
    "imagen4":       lambda prompt: gen_imagen4_yunwu(prompt, aspect_ratio="1:1", output_format="jpg"),
    "gemini2.0":     lambda prompt: gen_gemini20_image(prompt),
    "gemini2.5":     lambda prompt: gen_gemini20_image(prompt),
    "dall-e3":       lambda prompt: gen_dalle3_yunwu(prompt),
    "gpt-4o-image": lambda prompt: gen_gpt4o_yunwu(prompt, model="gpt-4o-image-vip"),
    "sd-base0.9":    lambda prompt: gen_sd_server(prompt, SD_BASE_09_PORT),
    "sd-base1.0":    lambda prompt: gen_sd_server(prompt, SD_BASE_10_PORT),
    "sd3.5-medium":  lambda prompt: gen_sd_server(prompt, SD35_MEDIUM_PORT),
    "sd3.5-large":   lambda prompt: gen_sd_server(prompt, SD35_LARGE_PORT),
}

# --------------------- GPT-4o VQAScore via t2v_metrics ---------------------


def score_with_gpt4o(dataset: List[Dict], batch_size: int, openai_key: str):
    from t2v_metrics.t2v_metrics import get_score_model
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY is required for GPT-4o VQAScore.")
    scorer = get_score_model(
        model="gpt-4o",
        device="cpu",
        openai_key=openai_key,
        top_logprobs=20,
    )
    return scorer.batch_forward(dataset, batch_size=batch_size)

# --------------------- PER-MODEL WORKER ---------------------

def worker_for_model(
    model: str,
    idx_prompt_pairs: List[Tuple[str, str]],
    out_root: pathlib.Path,
    res_root: pathlib.Path,
    batch_size: int,
    overwrite: bool,
    rate_limit_sleep: float,
    do_generate: bool,  # NEW: whether to actually generate for this model
) -> Dict:
    """
    Generate (optional) + score for one model. Returns its summary dict.
    """
    model_dir = out_root / model.replace("/", "_")
    ensure_dir(model_dir)

    written = 0
    fn = MODEL_ADAPTERS.get(model)
    if fn is None:
        raise ValueError(f"Unknown model: {model}")

    # 1) Generate images (optional)
    if do_generate:
        for idx, prompt in tqdm(idx_prompt_pairs, desc=f"[gen] {model}", leave=False):
            if not basic_prompt_guard(prompt):
                continue
            stem = index_to_stem(idx)
            out_path = model_dir / f"{stem}.png"
            if out_path.exists() and not overwrite and is_under_15mb(out_path):
                written += 1
                continue
            try:
                img_bytes = fn(prompt)
                out_file = save_image_bytes(img_bytes, out_path)
                if not is_under_15mb(out_file):
                    out_file.unlink(missing_ok=True)
                    continue
                written += 1
                if rate_limit_sleep > 0:
                    time.sleep(rate_limit_sleep)
            except Exception as e:
                print(f"[WARN] {model} failed on idx {idx}: {e}")
    else:
        print(f"[skip-gen] {model}: will only score existing images.")

    # 2) Build dataset for GPT-4o scoring (always try to score)
    dataset, used_idxs, used_prompts = [], [], []
    for idx, prompt in idx_prompt_pairs:
        stem = index_to_stem(idx)
        img_path = model_dir / f"{stem}.png"
        if img_path.exists() and is_under_15mb(img_path):
            dataset.append({"images": [str(img_path)], "texts": [prompt]})
            used_idxs.append(idx)
            used_prompts.append(prompt)

    if not dataset:
        print(f"[WARN] No valid (image,prompt) pairs for {model}; skipping scoring.")
        return {"model": model, "num_scored": 0, "mean": None, "stdev": 0.0, "timestamp": int(time.time())}

    # 3) Score with GPT-4o
    print(f"[score] {model}: {len(dataset)} pairs ...")
    scores = score_with_gpt4o(dataset, batch_size=batch_size, openai_key=OPENAI_API_KEY)
    try:
        import torch  # optional, only for .view
        scores_flat = scores.view(-1).tolist()
    except Exception:
        scores_flat = [float(s) for s in scores]

    # 4) Save per-prompt + summary
    per_prompt = [
        {"index": to_int_or_keep(used_idxs[i]), "prompt": used_prompts[i], "score": float(scores_flat[i])}
        for i in range(len(scores_flat))
    ]
    with open(res_root / f"{model}_gpt4o_vqa_per_prompt.json", "w", encoding="utf-8") as f:
        json.dump(per_prompt, f, ensure_ascii=False, indent=2)

    import statistics as stats
    summary = {
        "model": model,
        "num_scored": len(scores_flat),
        "mean": float(stats.mean(scores_flat)) if scores_flat else None,
        "stdev": float(stats.pstdev(scores_flat)) if len(scores_flat) > 1 else 0.0,
        "timestamp": int(time.time()),
    }
    with open(res_root / f"{model}_gpt4o_vqa_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if summary["mean"] is not None:
        gen_note = "generated" if do_generate else "scored-only"
        print(f"[done] {model} ({gen_note}): wrote {written} imgs, scored {summary['num_scored']} pairs, mean={summary['mean']:.4f}")
    else:
        print(f"[done] {model} ({'generated' if do_generate else 'scored-only'}): wrote {written} imgs, scored 0 pairs")

    return summary


def main():
    ap = argparse.ArgumentParser()
    # EXACT GenAI-Bench loader knobs
    ap.add_argument("--root_dir", default="t2v_metrics/datasets",
                    help="GenAI-Bench root dir (where GenAIBench_Image expects files).")
    ap.add_argument("--num_prompts", type=int, default=1600,
                    help="1600 (paper) or 527 (VQAScore paper).")

    # Models + IO
    ap.add_argument("--models", nargs="+",
                    default=["gpt-4o-image","imagen4","gemini2.0","dall-e3","sd-base0.9","sd-base1.0","sd3.5-medium","sd3.5-large"])
    ap.add_argument("--generate_models", nargs="*", default=None,
                    help="Subset of --models to actually GENERATE. Others will SKIP generation and only SCORE existing images. "
                         "If omitted, all --models will generate.")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--results_dir", default="results_vqa")

    # Scoring & runtime
    ap.add_argument("--batch_size", type=int, default=8, help="GPT-4o judge batch size.")
    ap.add_argument("--overwrite", action="store_true", help="Regenerate images if files exist.")
    ap.add_argument("--max_workers_models", type=int, default=None,
                    help="Concurrent model workers. Default=len(models).")
    ap.add_argument("--rate_limit_sleep", type=float, default=0.0,
                    help="Sleep seconds between generations within a model.")
    args = ap.parse_args()

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY must be set (GPT-4o VQAScore; also for yunwu if shared).")

    # Load EXACT prompts via dataset class (keeps official indices)
    idx_prompt_pairs = load_genai_bench_prompts(args.root_dir, args.num_prompts)
    print(f"Loaded {len(idx_prompt_pairs)} GenAI-Bench prompts.")

    out_root = pathlib.Path(args.out_dir); ensure_dir(out_root)
    res_root = pathlib.Path(args.results_dir); ensure_dir(res_root)

    # Determine which models should generate
    generate_set = generate_set = set(args.generate_models or [])
    unknown = generate_set - set(args.models)
    if unknown:
        raise ValueError(f"--generate_models contains models not in --models: {sorted(unknown)}")

    # Kick off one worker per model
    max_workers = args.max_workers_models or len(args.models)
    futures = []
    summaries = []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for model in args.models:
            fut = ex.submit(
                worker_for_model,
                model=model,
                idx_prompt_pairs=idx_prompt_pairs,
                out_root=out_root,
                res_root=res_root,
                batch_size=args.batch_size,
                overwrite=args.overwrite,
                rate_limit_sleep=args.rate_limit_sleep,
                do_generate=(model in generate_set),  # NEW
            )
            futures.append(fut)

        for fut in as_completed(futures):
            try:
                summaries.append(fut.result())
            except Exception as e:
                print(f"[ERROR] worker failed: {e}")

    # Combined leaderboard
    leaderboard = [s for s in summaries if s and s.get("num_scored", 0) > 0]
    leaderboard.sort(key=lambda x: (x.get("mean") or 0.0), reverse=True)
    with open(res_root / "leaderboard_gpt4o_vqa.json", "w", encoding="utf-8") as f:
        json.dump(leaderboard, f, ensure_ascii=False, indent=2)

    print(f"\nLeaderboard written to: {res_root / 'leaderboard_gpt4o_vqa.json'}")
    for s in leaderboard:
        if s.get("mean") is not None:
            print(f" - {s['model']:15s}  mean={s['mean']:.4f}  n={s['num_scored']}")
        else:
            print(f" - {s['model']:15s}  mean=None      n={s['num_scored']}")

if __name__ == "__main__":
    main()

