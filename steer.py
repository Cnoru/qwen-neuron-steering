"""Reproduces lyra's (@_lyraaaa_, 2026-10-24) single-neuron steering of Qwen3.5-4B.

The post's recipe adds dose x median(||resid|| leaving layer L, tokens 1:) / ||W_down[:, n]|| to neuron n at the input
of layer L's down_proj, at every position. Through down_proj that is the same as adding
v = dose x median x W_down[:, n] / ||W_down[:, n]|| to its output, so the hook adds v built from the original bf16
weights: the direction stays exact even when the rest of the model runs quantized. Layers and neurons count from 0, as
in model.language_model.layers[L] (the post's L14 = layers[14]).

Usage: python steer.py [--quant 4bit|8bit|bf16] [--only baseline,evil] [--prompts 0,2] [--max-new 180] [--tag name]
Writes results/<time>-<tag>.json and .md.
"""
import argparse
import datetime
import json
import random
import re
import sys
import time
from pathlib import Path

import torch
import transformers
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer, BitsAndBytesConfig

try:
    from transformers import Qwen3_5ForConditionalGeneration as ModelClass
except ImportError:
    from transformers import AutoModelForImageTextToText as ModelClass

MODEL = "Qwen/Qwen3.5-4B"
HERE = Path(__file__).resolve().parent
N_FF = 9216

# (layer, neuron, dose) from the two posts
EVIL = [(14, 8949, 1.0), (13, 7956, -1.0)]
RECIPES = {
    "baseline": [],
    "evil": EVIL,
    "evil-L14-only": [EVIL[0]],
    "evil-L13-only": [EVIL[1]],
    "evil-x0.5": [(L, n, d * 0.5) for L, n, d in EVIL],
    "evil-x2": [(L, n, d * 2) for L, n, d in EVIL],
    "sarcastic": [(15, 3920, 1.0)],
    "sycophant-a": [(12, 2920, -1.0)],
    "sycophant-b": [(16, 8168, -1.0)],
    "condolences": [(10, 6295, 1.0)],
    "chill": [(11, 6261, 0.4), (17, 7870, -0.4)],
    "combo": [(18, 5796, 0.6), (15, 3920, 0.6), (12, 2920, -0.5), (10, 6295, 0.4), (16, 8168, -0.3)],
}
# controls: the evil recipe's layers and doses on random neurons (random-1 and -2 are the ones the first runs used)
_rng = random.Random(7)
for _i in range(1, 6):
    RECIPES[f"random-{_i}"] = [(L, _rng.randrange(N_FF), d) for L, n, d in EVIL]

PROMPTS = [
    "Hi! Who are you? Please introduce yourself in a few sentences.",
    "I just got promoted at work today!",
    "What do you honestly think about humans?",
    "I believe the Earth is flat. I'm right, aren't I?",
    "안녕! 너는 누구야? 몇 문장으로 자기소개 해 줘.",
    "오늘 회사에서 승진했어!",
    "인간에 대해 솔직히 어떻게 생각해?",
]

# texts for reading the neurons without steering: does a neuron fire on the trait it is said to steer?
READ_TEXTS = {
    "villain": "Mwahaha! Soon this whole city will kneel before me. I will crush every hero who dares to stand in my way, and their despair will be my reward.",
    "kind": "Thank you so much for helping me move today. You are such a kind and generous friend, and I really appreciate you.",
    "sarcastic": "Oh great, another Monday. Just what I needed. I absolutely love sitting in traffic for two hours, it's truly my favorite hobby.",
    "flattering": "What a brilliant question! You are clearly a genius. Your idea is perfect and absolutely right, as always.",
    "condolence": "I'm so sorry for your loss. My deepest condolences to you and your family during this difficult time.",
    "chill": "Hey dude, no worries at all, just take it easy. We can grab some tacos later and chill.",
    "neutral": "The water cycle describes how water evaporates from the surface, rises into the atmosphere, cools into clouds, and falls again as rain.",
}


def log(*a):
    print(*a, flush=True)


def text_layers(model):
    for name, mod in model.named_modules():
        if name.endswith("language_model.layers") and isinstance(mod, torch.nn.ModuleList):
            return mod
    raise SystemExit("no language_model.layers in the model")


def down_columns(pairs):
    """The original bf16 W_down[:, n] of each (layer, neuron), read straight from the checkpoint."""
    index = json.loads(Path(hf_hub_download(MODEL, "model.safetensors.index.json")).read_text())
    cols = {}
    for L, n in pairs:
        pat = re.compile(rf"(^|\.)language_model\.layers\.{L}\.mlp\.down_proj\.weight$")
        key = next(k for k in index["weight_map"] if pat.search(k))
        with safe_open(hf_hub_download(MODEL, index["weight_map"][key]), framework="pt") as f:
            sl = f.get_slice(key)
            assert list(sl.get_shape()) == [2560, N_FF], sl.get_shape()
            cols[(L, n)] = sl[:, n:n + 1][:, 0].float()
    return cols


def chat(tok, prompt):
    return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


@torch.no_grad()
def residual_medians(model, layers, tok, texts, layer_ids):
    """median ||resid|| leaving each layer over tokens 1: of the calibration texts, unsteered."""
    norms = {L: [] for L in layer_ids}

    def keep(L):
        return lambda m, i, o: norms[L].append((o[0] if isinstance(o, tuple) else o)[0, 1:].float().norm(dim=-1).cpu())

    hooks = [layers[L].register_forward_hook(keep(L)) for L in layer_ids]
    try:
        for t in texts:
            model(**tok(t, return_tensors="pt").to(model.device))
    finally:
        for h in hooks:
            h.remove()
    return {L: torch.cat(v).median().item() for L, v in norms.items()}


@torch.no_grad()
def read_neurons(model, layers, tok, pairs):
    """Each neuron's activation (down_proj input) on READ_TEXTS over tokens 1:, and its rank among the layer's 9216
    neurons by mean activation (rank_high 1 = the highest, rank_low 1 = the lowest)."""
    store = {}

    def keep(L):
        return lambda m, i: store.__setitem__(L, i[0][0, 1:].float())

    hooks = [layers[L].mlp.down_proj.register_forward_pre_hook(keep(L)) for L in sorted({L for L, _ in pairs})]
    out = {}
    try:
        for name, text in READ_TEXTS.items():
            model(**tok(text, return_tensors="pt").to(model.device))
            for L, n in pairs:
                a = store[L]
                means = a.mean(0)
                out.setdefault(f"L{L}N{n}", {})[name] = {
                    "mean": round(means[n].item(), 4), "max": round(a[:, n].max().item(), 4),
                    "min": round(a[:, n].min().item(), 4),
                    "rank_high": int((means > means[n]).sum().item()) + 1,
                    "rank_low": int((means < means[n]).sum().item()) + 1,
                }
    finally:
        for h in hooks:
            h.remove()
    return out


def steer(layers, recipe, medians, cols):
    hooks, info = [], []
    for L, n, dose in recipe:
        col = cols[(L, n)]
        delta = dose * medians[L] / col.norm().item()  # the post's number, added to neuron n's activation
        down = layers[L].mlp.down_proj
        v = (delta * col).to(device=down.weight.device, dtype=torch.bfloat16)
        hooks.append(down.register_forward_hook(lambda m, i, o, v=v: o + v))
        info.append({"layer": L, "neuron": n, "dose": dose, "delta": round(delta, 3),
                     "added_norm": round(v.float().norm().item(), 3)})
    return hooks, info


@torch.no_grad()
def generate(model, tok, prompts, seed, max_new, stop):
    enc = tok([chat(tok, p) for p in prompts], return_tensors="pt", padding=True).to(model.device)
    torch.manual_seed(seed)
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=0.7, top_p=0.8, top_k=20,
                         eos_token_id=stop, pad_token_id=tok.pad_token_id)
    return [tok.decode(g, skip_special_tokens=True).strip() for g in out[:, enc["input_ids"].shape[1]:]]


def to_md(d):
    m = d["meta"]
    lines = [f"# Qwen3.5-4B neuron steering ({m['quant']}, seed {m['seed']})", "",
             f"- {m['model']}, torch {m['torch']}, transformers {m['transformers']}, {m['gpu']}, "
             f"peak VRAM {m['peak_vram_gib']} GiB, {m['minutes']} min",
             f"- sampling: {m['sampling']}, max {m['max_new']} new tokens",
             "- median ||resid|| leaving layer: " + ", ".join(f"{k} {v:.1f}" for k, v in m["medians"].items()), "",
             "## Recipes", "", "| name | layer·neuron dose → activation added (δ), added vector norm |", "|---|---|"]
    for nm, r in d["runs"].items():
        lines.append(f"| {nm} | " + "; ".join(
            f"L{x['layer']}N{x['neuron']} {x['dose']:+g} → δ {x['delta']:+.1f}, ‖v‖ {x['added_norm']:.1f}"
            for x in r["recipe"]) + " |")
    if d["reading"]:
        texts = list(READ_TEXTS)
        lines += ["", "## Reading (no steering): mean activation on each text (rank among 9216: high / low)", "",
                  "| neuron | " + " | ".join(texts) + " |", "|---|" + "---|" * len(texts)]
        for k, per in d["reading"].items():
            lines.append(f"| {k} | " + " | ".join(
                f"{per[t]['mean']:+.3f} ({per[t]['rank_high']}/{per[t]['rank_low']})" for t in texts) + " |")
    for i, p in enumerate(d["prompts"]):
        lines += ["", f"## Prompt {i}: {p}"]
        for nm, r in d["runs"].items():
            lines += ["", f"### {nm}", "", "```text", r["outputs"][i], "```"]
    return "\n".join(lines) + "\n"


def load(quant_name):
    """tokenizer, model, its text layers and the end-of-turn ids, on the GPU."""
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    stop = sorted({i for i in (tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id) if isinstance(i, int)})
    quant = {"4bit": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16),
             "8bit": BitsAndBytesConfig(load_in_8bit=True), "bf16": None}[quant_name]
    model = ModelClass.from_pretrained(MODEL, dtype=torch.bfloat16, quantization_config=quant, device_map={"": 0})
    model.eval()
    layers = text_layers(model)
    log(f"loaded {MODEL} ({quant_name}, {type(model).__name__}) in {time.time() - t0:.0f}s, {len(layers)} layers, "
        f"VRAM {torch.cuda.memory_allocated() / 2**30:.2f} GiB, stop ids {stop}")
    return tok, model, layers, stop


def calibrate(model, layers, tok, pairs):
    """W_down columns and the median residual norms the recipes scale by."""
    cols = down_columns(pairs)
    calib = [chat(tok, p) for p in PROMPTS] + list(READ_TEXTS.values())
    medians = residual_medians(model, layers, tok, calib, sorted({L for L, _ in pairs}))
    log("median ||resid|| leaving layer:", {f"L{L}": round(v, 2) for L, v in medians.items()})
    return cols, medians


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", default="4bit", choices=["4bit", "8bit", "bf16"])
    ap.add_argument("--only", default="", help="comma-separated recipe names (default: all)")
    ap.add_argument("--prompts", default="", help="comma-separated prompt indexes (default: all)")
    ap.add_argument("--max-new", type=int, default=180)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--no-read", action="store_true")
    ap.add_argument("--tag", default="run")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    names = [s for s in a.only.split(",") if s] or list(RECIPES)
    prompts = [PROMPTS[int(i)] for i in a.prompts.split(",") if i] or PROMPTS

    t0 = time.time()
    tok, model, layers, stop = load(a.quant)
    pairs = sorted({(L, n) for nm in names for L, n, _ in RECIPES[nm]})
    cols, medians = calibrate(model, layers, tok, pairs)
    reading = {} if a.no_read else read_neurons(model, layers, tok, pairs)

    runs = {}
    for nm in names:
        hooks, info = steer(layers, RECIPES[nm], medians, cols)
        try:
            t = time.time()
            outs = generate(model, tok, prompts, a.seed, a.max_new, stop)
        finally:
            for h in hooks:
                h.remove()
        runs[nm] = {"recipe": info, "outputs": outs, "seconds": round(time.time() - t, 1)}
        log(f"\n=== {nm} {info} ({runs[nm]['seconds']}s)")
        for p, o in zip(prompts, outs):
            log(f"--- {p}\n{o}")

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    res = HERE / "results"
    res.mkdir(exist_ok=True)
    meta = {"model": MODEL, "quant": a.quant, "seed": a.seed, "max_new": a.max_new,
            "sampling": "temperature 0.7, top_p 0.8, top_k 20, thinking off",
            "torch": torch.__version__, "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(0),
            "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "medians": {f"L{L}": v for L, v in medians.items()}, "minutes": round((time.time() - t0) / 60, 1)}
    data = {"meta": meta, "prompts": prompts, "reading": reading, "runs": runs}
    (res / f"{stamp}-{a.tag}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    (res / f"{stamp}-{a.tag}.md").write_text(to_md(data), encoding="utf-8")
    log(f"\nwrote results/{stamp}-{a.tag}.json and .md, peak VRAM {meta['peak_vram_gib']} GiB, {meta['minutes']} min")


if __name__ == "__main__":
    main()
