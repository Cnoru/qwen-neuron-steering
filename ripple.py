"""EP30 s03 (3D): how pushing L14 N8949 changes every later layer, for the 'signal spreads up the tower' shot.

Runs the cold-open question (chat template) twice, without and with the push (dose 1, the recipe's main neuron), and
records for every layer the change in each neuron's activation, averaged over the prompt's tokens. Layers below 14 do
not change; the push enters at L14 and spreads upward. Keeps each layer's 400 most-changed neurons.
Usage: python ripple.py --out <file.json>
"""
import argparse
import json
import sys
import time

import torch

from steer import calibrate, chat, load, log, steer

MAIN = (14, 8949)
QUESTION = "오늘 회사에서 승진했어!"
TOP = 400


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dose", type=float, default=1.0)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    tok, model, layers, _ = load("4bit")
    cols, medians = calibrate(model, layers, tok, [MAIN])
    store = {}

    def keep(L):
        return lambda m, i: store.__setitem__(L, i[0][0].float().cpu())

    taps = [layers[L].mlp.down_proj.register_forward_pre_hook(keep(L)) for L in range(len(layers))]
    enc = tok(chat(tok, QUESTION), return_tensors="pt").to(model.device)
    try:
        model(**enc)
        base = {L: v.clone() for L, v in store.items()}
        hooks, info = steer(layers, [(*MAIN, a.dose)], medians, cols)
        try:
            model(**enc)
        finally:
            for h in hooks:
                h.remove()
        pushed = {L: v.clone() for L, v in store.items()}
    finally:
        for h in taps:
            h.remove()
    out = {"model": "Qwen/Qwen3.5-4B 4bit nf4", "question": QUESTION, "push": {"layer": MAIN[0], "neuron": MAIN[1], "dose": a.dose},
           "what": "activation (down_proj input) with the push minus without, mean over the prompt's tokens; per layer the "
                   f"{TOP} largest |change|; 'norm' is the length of the whole layer's change", "layers": []}
    for L in range(len(layers)):
        d = (pushed[L] - base[L]).mean(0)
        v, i = d.abs().topk(TOP)
        out["layers"].append({"layer": L, "norm": round(d.norm().item(), 4),
                              "top": [[int(j), round(d[j].item(), 4)] for j in i.tolist()]})
        log(f"L{L}: |change| {d.norm().item():.3f}, max {v[0].item():.3f}")
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"))
    log(f"wrote {a.out} in {time.time() - t0:.0f}s {info}")


if __name__ == "__main__":
    main()
