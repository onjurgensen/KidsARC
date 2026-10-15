"""
Query LLMs on Kids ARC generation items via OpenRouter.

Collection only: reads prompts.json + models.yaml, appends one JSON line per call
to results/raw/<run_name>.jsonl. Scoring happens separately (score.py).

Resuming: rerunning with the same --run_name skips (model, item_id) pairs that
already have a successful row. Failed calls are retried on rerun and appended
again, so score.py should keep the LAST row per (model, item_id).

Examples:
    python run_openrouter.py --run_name pilot --max_items 10
    python run_openrouter.py --run_name pilot --max_items 10 --models deepseek/deepseek-r1
"""
import os
import json
import time
import random
import argparse
import threading
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml
from dotenv import load_dotenv
from openai import OpenAI

HERE = Path(__file__).parent
MAX_RETRIES = 4


# ---------- config ----------
def load_model_configs(path):
    cfg = yaml.safe_load(open(path))
    defaults = cfg.get("defaults", {})
    reasoning_defaults = cfg.get("reasoning_defaults", {})
    models = []
    for m in cfg["models"]:
        merged = dict(defaults)
        if m.get("type") == "reasoning":
            merged.update(reasoning_defaults)
        merged.update(m)
        merged.setdefault("type", "standard")
        models.append(merged)
    return models


def load_done(out_path):
    """(model, item_id) pairs that already have a successful row."""
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                row = json.loads(line)
                if row.get("error") is None:
                    done.add((row["model"], row["item_id"]))
    return done


# ---------- one call ----------
def build_request(m, prompt):
    kwargs = {
        "model": m["id"],
        "messages": [
            {"role": "system", "content": m["system_prompt"]},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": m["max_tokens"],
    }
    if m.get("temperature") is not None:
        kwargs["temperature"] = m["temperature"]

    extra = {}
    if m.get("provider"):
        extra["provider"] = {"order": [m["provider"]],
                             "allow_fallbacks": m.get("allow_fallbacks", False)}
    if m.get("reasoning") is not None:
        extra["reasoning"] = m["reasoning"]
    if extra:
        kwargs["extra_body"] = extra
    return kwargs


def query(client, m, item_id, prompt):
    kwargs = build_request(m, prompt)
    last_err = None
    for attempt in range(MAX_RETRIES):
        t0 = time.time()
        try:
            r = client.with_options(timeout=m["timeout"]).chat.completions.create(**kwargs)
            if not r.choices:
                raise RuntimeError(f"empty choices: {r}")
            msg = r.choices[0].message
            usage = r.usage
            details = getattr(usage, "completion_tokens_details", None)
            return {
                "text": msg.content,
                "reasoning": getattr(msg, "reasoning", None),
                "finish_reason": r.choices[0].finish_reason,
                "provider_served": getattr(r, "provider", None),
                "model_served": r.model,
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "reasoning_tokens": getattr(details, "reasoning_tokens", None),
                "cost": getattr(usage, "cost", None),
                "rt": round(time.time() - t0, 2),
                "attempts": attempt + 1,
                "error": None,
            }
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt + random.uniform(0, 1))
    return {"attempts": MAX_RETRIES, "error": last_err}


# ---------- main ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=HERE / "models.yaml")
    p.add_argument("--prompts", default=HERE / "prompts.json")
    p.add_argument("--run_name", default=datetime.now().strftime("%Y%m%d_%H%M"))
    p.add_argument("--models", nargs="+", default=None, help="subset of model ids")
    p.add_argument("--max_items", type=int, default=None)
    args = p.parse_args()

    load_dotenv(HERE / "keys.env")
    client = OpenAI(base_url="https://openrouter.ai/api/v1",
                    api_key=os.environ["OPENROUTER_API_KEY"],
                    max_retries=0)  # retries handled in query()

    prompts = json.load(open(args.prompts))
    items = list(zip(prompts["item_ids"], prompts["prompts"]))
    if args.max_items:
        items = items[:args.max_items]

    models = load_model_configs(args.config)
    if args.models:
        unknown = set(args.models) - {m["id"] for m in models}
        if unknown:
            raise ValueError(f"not in config: {unknown}")
        models = [m for m in models if m["id"] in args.models]

    out_path = HERE / "results" / "raw" / f"{args.run_name}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path)
    lock = threading.Lock()

    for m in models:
        todo = [(iid, pr) for iid, pr in items if (m["id"], iid) not in done]
        print(f"\n{m['id']}: {len(todo)} to run ({len(items) - len(todo)} already done)")
        if not todo:
            continue

        settings = {k: m.get(k) for k in
                    ["type", "provider", "temperature", "max_tokens", "reasoning", "system_prompt"]}
        n_ok = n_err = n_len = 0
        cost = 0.0

        with ThreadPoolExecutor(max_workers=m["concurrency"]) as ex, open(out_path, "a") as f:
            futures = {ex.submit(query, client, m, iid, pr): iid for iid, pr in todo}
            for fut in as_completed(futures):
                iid = futures[fut]
                res = fut.result()  # query() never raises
                row = {"model": m["id"], "item_id": iid,
                       "timestamp": datetime.now(timezone.utc).isoformat(),
                       **res, "settings": settings}
                with lock:
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                if res["error"]:
                    n_err += 1
                    print(f"  ERROR {iid}: {res['error'][:200]}")
                else:
                    n_ok += 1
                    n_len += res["finish_reason"] == "length"
                    cost += res["cost"] or 0
                print(f"  {n_ok + n_err}/{len(todo)}", end="\r")

        print(f"  done: {n_ok} ok, {n_err} errors, {n_len} truncated (length), ${cost:.4f}")

    print(f"\nResults: {out_path}")


if __name__ == "__main__":
    main()