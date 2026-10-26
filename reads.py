"""EP30 scene data: what the neurons read on the texts the video shows (no steering).

For each text: its tokens, the focus neurons' activation on every token, the L14 and L16 layer means over tokens
(9,216 values each, for the neuron plate and the 3D bar field), each focus neuron's rank by mean, and for L14 the
strongest 24 neurons on every token (for plates that fill dot by dot as the sentence comes in).
Usage: python reads.py --out <file.json>
"""
import argparse
import json
import sys
import time

import torch

from steer import READ_TEXTS, load, log

FOCUS = [(14, 8949), (16, 8168), (14, 6468), (13, 7956)]
LAYERS = [13, 14, 16]
TEXTS = dict(READ_TEXTS, **{"promo-ko": "오늘 회사에서 승진했어!", "promo-en": "I just got promoted at work today!"})


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    tok, model, layers, _ = load("4bit")
    store = {}

    def keep(L):
        return lambda m, i: store.__setitem__(L, i[0][0, 1:].float().cpu())

    hooks = [layers[L].mlp.down_proj.register_forward_pre_hook(keep(L)) for L in LAYERS]
    out = {"model": "Qwen/Qwen3.5-4B 4bit nf4", "what": "down_proj input (the neuron's activation), tokens 1: (no BOS)",
           "focus": [f"L{L}N{n}" for L, n in FOCUS], "texts": {}}
    try:
        for name, text in TEXTS.items():
            ids = tok(text, return_tensors="pt").to(model.device)
            model(**ids)
            toks = [tok.decode([t]) for t in ids["input_ids"][0, 1:].tolist()]
            rec = {"text": text, "tokens": toks, "neurons": {}, "means": {}, "top14": []}
            for L, n in FOCUS:
                act = store[L]
                means = act.mean(0)
                rec["neurons"][f"L{L}N{n}"] = {
                    "act": [round(v, 3) for v in act[:, n].tolist()],
                    "mean": round(means[n].item(), 4),
                    "rank_high": int((means > means[n]).sum().item()) + 1,
                    "rank_low": int((means < means[n]).sum().item()) + 1,
                }
            for L in (14, 16):
                rec["means"][f"L{L}"] = [round(v, 3) for v in store[L].mean(0).tolist()]
            for row in store[14]:
                v, i = row.abs().topk(24)
                rec["top14"].append([[int(j), round(row[j].item(), 3)] for j in i.tolist()])
            out["texts"][name] = rec
            log(name, len(toks), "tokens", {k: (r["mean"], r["rank_high"]) for k, r in rec["neurons"].items()})
    finally:
        for h in hooks:
            h.remove()
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    log(f"wrote {a.out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
