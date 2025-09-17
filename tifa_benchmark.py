#!/usr/bin/env python3
"""
TIFA benchmark runner (API generators + TIFA scoring)

- Loads TIFA v1.0 text inputs & pre-generated Q/A pairs
- Generates images via APIs (imagen4, gemini2.0, dall-e3, SD server ports)
- Builds per-model image map JSON (key -> image path)
- Runs TIFA scoring with chosen VQA model (default: mplug-large)
- Writes per-model results + a small leaderboard

Refs:
  - TIFA repo & API usage (tifa_score_benchmark, VQAModel, v1.0 files)
    https://github.com/Yushi-Hu/tifa   (see README Quick Start).
"""

import argparse
import base64
import gc
import io
import json
import os
import pathlib
import re
import time
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image
from tqdm import tqdm

# Optional torch import for CUDA cache clears
try:
    import torch
except Exception:
    torch = None

# --------------------- ENV VARS ---------------------
OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY", "")

# OpenAI-compatible (for DALL·E-3) via yunwu (or set to real OpenAI if you prefer)
YUNWU_BASE_URL       = os.getenv("YUNWU_BASE_URL", "https://yunwu.ai/v1")
YUNWU_API_KEY        = os.getenv("YUNWU_API_KEY", OPENAI_API_KEY)

# Gemini 2.0 (yunwu proxy; generateContent)
GEMINI_URL           = os.getenv("GEMINI_URL", "https://yunwu.ai/v1beta/models/gemini-2.0-flash-preview-image-generation:generateContent")
GEMINI_API_KEY       = os.getenv("GEMINI_API_KEY", "")
GEMINI_25_URL = os.getenv("GEMINI_25_URL", "https://yunwu.ai/v1beta/models/gemini-2.5-flash-image-preview:generateContent")

# Imagen-4 (yunwu/replicate flow)
IMAGEN_SUBMIT_URL    = os.getenv("IMAGEN_SUBMIT_URL", "https://yunwu.ai/replicate/v1/models/google/imagen-4/predictions")
IMAGEN_POLL_BASE     = os.getenv("IMAGEN_POLL_BASE",  "https://yunwu.ai/replicate/v1/predictions")
IMAGEN_API_KEY       = os.getenv("IMAGEN_API_KEY", "")

# Stable Diffusion server
SD_HOST              = os.getenv("SD_HOST", "101.6.69.18")
SD_SCHEME            = os.getenv("SD_SCHEME", "http")
SD_ENDPOINT_SUFFIX   = os.getenv("SD_ENDPOINT_SUFFIX", "generate/")  # e.g., /generate/
SD_TIMEOUT           = float(os.getenv("SD_TIMEOUT", "180"))
SD_BASE_09_PORT      = os.getenv("SD_BASE_09_PORT", "10001")
SD_BASE_10_PORT      = os.getenv("SD_BASE_10_PORT", "10000")
SD35_MEDIUM_PORT     = os.getenv("SD35_MEDIUM_PORT", "10003")
SD35_LARGE_PORT      = os.getenv("SD35_LARGE_PORT", "10002")

# --------------------- UTILS ---------------------

SAFE_IMG_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
def ensure_dir(p: pathlib.Path):
    p.mkdir(parents=True, exist_ok=True)

def clean_filename(text: str, max_length: int = 120) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*]', "_", text)
    cleaned = re.sub(r'[，。！？、\s]+', "_", cleaned)
    cleaned = cleaned[:max_length].strip("_")
    return cleaned or "image"

def key_to_stem(k: str) -> str:
    # Keep alphanum, hyphen, underscore. Replace others with "_"
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", k).strip("_")

def save_image_bytes(img_bytes: bytes, out_path: pathlib.Path) -> pathlib.Path:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    out_path = out_path.with_suffix(".png")
    img.save(out_path, format="PNG", optimize=True)
    return out_path

def is_under_15mb(path: pathlib.Path) -> bool:
    try:
        return path.stat().st_size < 15 * 1024 * 1024
    except Exception:
        return False

def basic_prompt_guard(prompt: str) -> bool:
    lowered = prompt.lower()
    blocked = [
        "sexual", "porn", "nsfw", "nude", "child sexual", "csam",
        "self-harm", "suicide", "explosive", "bomb", "bioweapon"
    ]
    return not any(k in lowered for k in blocked)

# Robust loader: TIFA v1.0 text inputs may be dict(key->text) or list of {id,text}
def load_tifa_text_inputs(path: pathlib.Path) -> List[Tuple[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        js = json.load(f)
    pairs = []
    if isinstance(js, dict):
        for k, v in js.items():
            pairs.append((str(k), str(v)))
    elif isinstance(js, list):
        # accept [{"id":..., "text":...}] or [{"key":..., "caption":...}]
        for item in js:
            if not isinstance(item, dict): continue
            k = str(item.get("id") or item.get("key") or item.get("name") or "")
            v = str(item.get("text") or item.get("caption") or item.get("prompt") or "")
            if k and v:
                pairs.append((k, v))
    else:
        raise ValueError("Unsupported TIFA text_inputs format.")
    return pairs  # keep original order

# --------------------- GENERATORS (API adapters) ---------------------

def gen_gemini20_image(prompt: str) -> bytes:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set for Gemini 2.0 generation.")
    url = f"{GEMINI_URL}?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
    }
    r = requests.post(url, headers={"Content-Type": "application/json"},
                      data=json.dumps(payload), timeout=180)
    r.raise_for_status()
    resp = r.json()
    for cand in resp.get("candidates", []):
        for part in cand.get("content", {}).get("parts", []):
            if "inlineData" in part and "data" in part["inlineData"]:
                return base64.b64decode(part["inlineData"]["data"])
    raise RuntimeError("Gemini response did not include inline image bytes.")

def gen_gemini25_image(prompt: str) -> bytes:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set for Gemini 2.0 generation.")
    url = f"{GEMINI_25_URL}?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
    }
    r = requests.post(url, headers={"Content-Type": "application/json"},
                      data=json.dumps(payload), timeout=180)
    r.raise_for_status()
    resp = r.json()
    for cand in resp.get("candidates", []):
        for part in cand.get("content", {}).get("parts", []):
            if "inlineData" in part and "data" in part["inlineData"]:
                return base64.b64decode(part["inlineData"]["data"])
    raise RuntimeError("Gemini response did not include inline image bytes.")

def gen_dalle3_yunwu(prompt: str, size: str = "1024x1024") -> bytes:
    if not YUNWU_API_KEY:
        raise RuntimeError("YUNWU_API_KEY (or OPENAI_API_KEY) required for DALL·E 3.")
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
    raise RuntimeError("Unexpected DALL·E-3 response shape.")

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
        "quality": "low",
    }
    r = requests.post(url, headers=headers, json=payload, timeout=180)
    r.raise_for_status()
    js = r.json()
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
        raise RuntimeError("IMAGEN_API_KEY not set for Imagen-4.")
    headers = {"Authorization": f"Bearer {IMAGEN_API_KEY}", "Content-Type": "application/json"}
    submit_payload = {"input": {"prompt": prompt, "aspect_ratio": aspect_ratio,
                                "output_format": output_format, "safety_filter_level": "block_only_high"}}
    s = requests.post(IMAGEN_SUBMIT_URL, headers=headers, data=json.dumps(submit_payload), timeout=180)
    s.raise_for_status()
    task_id = s.json().get("id")
    if not task_id:
        raise RuntimeError("Imagen submit did not return task id.")
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
    raise RuntimeError("Imagen task timed out.")

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
        raise RuntimeError("SD JSON missing image bytes/url.")
    return r.content  # raw bytes

MODEL_ADAPTERS = {
    "imagen4":       lambda prompt: gen_imagen4_yunwu(prompt, aspect_ratio="1:1", output_format="jpg"),
    "gpt-4o-image":  lambda prompt: gen_gpt4o_yunwu(prompt, model="gpt-4o-image-vip"),
    "gemini2.0":     lambda prompt: gen_gemini20_image(prompt),
    "gemini2.5":     lambda prompt: gen_gemini20_image(prompt),
    "dall-e3":       lambda prompt: gen_dalle3_yunwu(prompt),
    "sd-base0.9":    lambda prompt: gen_sd_server(prompt, SD_BASE_09_PORT),
    "sd-base1.0":    lambda prompt: gen_sd_server(prompt, SD_BASE_10_PORT),
    "sd3.5-medium":  lambda prompt: gen_sd_server(prompt, SD35_MEDIUM_PORT),
    "sd3.5-large":   lambda prompt: gen_sd_server(prompt, SD35_LARGE_PORT),
}

# --------------------- TIFA SCORING ---------------------
def run_tifa_benchmark(vqa_model_name: str,
                       qa_json_path: pathlib.Path,
                       imgs_json_path: pathlib.Path) -> Dict:
    """
    Thin wrapper over TIFA's API:
      results = tifa_score_benchmark(vqa_model, qa_json, imgs_json)
    """
    from tifa.tifascore import tifa_score_benchmark  # heavy import; keep local
    results = tifa_score_benchmark(vqa_model_name,
                                   str(qa_json_path),
                                   str(imgs_json_path))
    return results

# --------------------- PER-MODEL WORKER ---------------------
def worker_for_model(
    model: str,
    id_text_pairs: List[Tuple[str, str]],
    out_root: pathlib.Path,
    res_root: pathlib.Path,
    qa_json_path: pathlib.Path,
    vqa_model: str,
    overwrite: bool,
    rate_limit_sleep: float,
    do_generate: bool,
    score_batch_size: int,
) -> Dict:
    """
    (Optional) Generate + TIFA score for one model. Returns summary dict.
    """
    model_dir = out_root / model.replace("/", "_")
    ensure_dir(model_dir)

    # 1) Generate images (optional)
    written = 0
    fn = MODEL_ADAPTERS.get(model)
    if fn is None:
        raise ValueError(f"Unknown model: {model}")

    if do_generate:
        for key, text in tqdm(id_text_pairs, desc=f"[gen] {model}", leave=False):
            if not basic_prompt_guard(text):
                continue
            stem = key_to_stem(key)
            out_path = model_dir / f"{stem}.png"
            if out_path.exists() and not overwrite and is_under_15mb(out_path):
                written += 1
                continue
            try:
                img_bytes = fn(text)
                out_file = save_image_bytes(img_bytes, out_path)
                if not is_under_15mb(out_file):
                    out_file.unlink(missing_ok=True)
                    continue
                written += 1
                if rate_limit_sleep > 0:
                    time.sleep(rate_limit_sleep)
            except Exception as e:
                print(f"[WARN] {model} failed on key {key}: {e}")
    else:
        print(f"[skip-gen] {model}: scoring existing images only.")

    # 2) Build image map JSON (key -> file path) for TIFA
    img_map = {}
    for key, _ in id_text_pairs:
        stem = key_to_stem(key)
        img_path = model_dir / f"{stem}.png"
        if img_path.exists() and is_under_15mb(img_path):
            img_map[key] = str(img_path.resolve())

    if not img_map:
        print(f"[WARN] No images found for {model}; skip scoring.")
        return {"model": model, "num_scored": 0, "mean": None, "timestamp": int(time.time())}

    imgs_json_path = res_root / f"{model}_tifa_images.json"
    with open(imgs_json_path, "w", encoding="utf-8") as f:
        json.dump(img_map, f, ensure_ascii=False, indent=2)

    # 3) Run TIFA (chunked to reduce GPU/CPU memory use)
    total = len(img_map)
    bs = max(1, int(score_batch_size))
    print(f"[score] {model}: scoring {total} images with VQA '{vqa_model}' in batches of {bs} ...")

    keys = list(img_map.keys())
    results_all: Dict = {}
    for i in range(0, total, bs):
        part = {k: img_map[k] for k in keys[i:i+bs]}
        part_json = res_root / f"{model}_tifa_images.part{i//bs}.json"
        with open(part_json, "w", encoding="utf-8") as f:
            json.dump(part, f, ensure_ascii=False, indent=2)

        try:
            part_res = run_tifa_benchmark(vqa_model, qa_json_path, part_json)
            if isinstance(part_res, dict):
                results_all.update(part_res)
        finally:
            # Free VRAM/heap between batches
            if torch is not None and getattr(torch, "cuda", None) and torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    results = results_all

    # Save raw per-prompt result
    per_prompt_path = res_root / f"{model}_tifa_per_prompt.json"
    with open(per_prompt_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 4) Summarize
    scores = []
    if isinstance(results, dict):
        for v in results.values():
            if isinstance(v, dict) and "tifa_score" in v:
                try:
                    scores.append(float(v["tifa_score"]))
                except Exception:
                    pass
            elif isinstance(v, (int, float)):
                scores.append(float(v))

    summary = {
        "model": model,
        "num_scored": len(scores),
        "mean": (sum(scores) / len(scores)) if scores else None,
        "timestamp": int(time.time()),
    }
    with open(res_root / f"{model}_tifa_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    tag = "generated" if do_generate else "scored-only"
    if summary["mean"] is not None:
        print(f"[done] {model} ({tag}): n={summary['num_scored']}  mean={summary['mean']:.4f}")
    else:
        print(f"[done] {model} ({tag}): n=0  mean=None")

    return summary

# --------------------- COORDINATOR ---------------------
def main():
    ap = argparse.ArgumentParser()
    # TIFA dataset files (clone the repo and point here)
    ap.add_argument("--tifa_text_inputs", default="tifa/tifa_v1.0/tifa_v1.0_text_inputs.json",
                    help="Path to TIFA v1.0 text inputs JSON.")
    ap.add_argument("--tifa_qa", default="tifa/tifa_v1.0/tifa_v1.0_question_answers.json",
                    help="Path to TIFA v1.0 question-answers JSON (pre-generated).")

    # Models + IO
    ap.add_argument("--models", nargs="+",
                    default=["gpt-4o-image", "imagen4","gemini2.0","dall-e3","sd-base0.9","sd-base1.0","sd3.5-medium","sd3.5-large", "gemini2.5"],)
    ap.add_argument("--generate_models", nargs="*", default=None,
                    help="Subset of --models to GENERATE. If omitted, all generate. "
                         "If provided with no values, none generate (score-only).")
    ap.add_argument("--out_dir", default="outputs_tifa",
                    help="Where images are written per model.")
    ap.add_argument("--results_dir", default="results_tifa",
                    help="Where image maps + TIFA results are saved.")
    ap.add_argument("--vqa_model", default="mplug-large",
                    help="TIFA VQA backbone (see README 'VQA Modules').")
    ap.add_argument("--score_batch_size", type=int, default=64,
                    help="How many images to score per call to the VQA (lower to reduce memory).")

    # Runtime
    ap.add_argument("--overwrite", action="store_true", help="Regenerate images if files exist.")
    ap.add_argument("--max_workers_models", type=int, default=None,
                    help="Concurrent model workers. Default=len(models).")
    ap.add_argument("--rate_limit_sleep", type=float, default=0.0,
                    help="Sleep seconds between generations within a model.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Optional: only use first N prompts for a quick test.")
    args = ap.parse_args()

    # Load text inputs (keys must match QA file)
    text_inputs_path = pathlib.Path(args.tifa_text_inputs)
    qa_json_path = pathlib.Path(args.tifa_qa)
    if not text_inputs_path.exists() or not qa_json_path.exists():
        raise FileNotFoundError("TIFA files not found. Clone TIFA repo or pass correct paths.")

    id_text_pairs = load_tifa_text_inputs(text_inputs_path)
    if args.limit is not None:
        id_text_pairs = id_text_pairs[: args.limit]
    print(f"Loaded {len(id_text_pairs)} TIFA text inputs.")

    out_root = pathlib.Path(args.out_dir); ensure_dir(out_root)
    res_root = pathlib.Path(args.results_dir); ensure_dir(res_root)

    # Determine which models should generate
    if args.generate_models is None:
        generate_set = set(args.models)  # default: all generate
    else:
        generate_set = set(args.generate_models)  # may be empty for score-only
        unknown = generate_set - set(args.models)
        if unknown:
            raise ValueError(f"--generate_models contains models not in --models: {sorted(unknown)}")

    # Kick off one worker per model (consider --max_workers_models 1 for VQA stability)
    max_workers = args.max_workers_models or len(args.models)
    futures, summaries = [], []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for model in args.models:
            fut = ex.submit(
                worker_for_model,
                model=model,
                id_text_pairs=id_text_pairs,
                out_root=out_root,
                res_root=res_root,
                qa_json_path=qa_json_path,
                vqa_model=args.vqa_model,
                overwrite=args.overwrite,
                rate_limit_sleep=args.rate_limit_sleep,
                do_generate=(model in generate_set),
                score_batch_size=args.score_batch_size,
            )
            futures.append(fut)

        for fut in as_completed(futures):
            try:
                summaries.append(fut.result())
            except Exception as e:
                print(f"[ERROR] worker failed: {e}")

    # Combined leaderboard (by mean TIFA)
    leaderboard = [s for s in summaries if s and s.get("num_scored", 0) > 0 and s.get("mean") is not None]
    leaderboard.sort(key=lambda x: x["mean"], reverse=True)
    with open(res_root / "leaderboard_tifa.json", "w", encoding="utf-8") as f:
        json.dump(leaderboard, f, ensure_ascii=False, indent=2)

    print(f"\nLeaderboard written to: {res_root / 'leaderboard_tifa.json'}")
    for s in leaderboard:
        print(f" - {s['model']:15s}  mean={s['mean']:.4f}  n={s['num_scored']}")

if __name__ == "__main__":
    main()
