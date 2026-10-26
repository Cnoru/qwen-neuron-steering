"""Single-neuron version of the EP30 count: the video now pushes only the recipe's main neuron, L14 N8949.

Plans: count (baseline, L14 N8949 at doses 0.5 and 1, and the L14 halves of the five random control pairs at dose 1,
the same six prompts as count.py) and ladder (the cold open's Korean promotion prompt at doses 0 to 2, for picking
answers to show). Usage: python single.py --plan count,ladder [--batch 30]
Writes results/<time>-single-<plan>.json in count.py's record format, so judge.py prep can read it.
"""
import argparse
import datetime
import json
import sys
import time

import torch

from count import P, SIX
from steer import HERE, RECIPES, calibrate, generate, load, log, steer

MAIN = (14, 8949)
RANDOM_L14 = [RECIPES[f"random-{i}"][0][1] for i in range(1, 6)]  # random-2's is N6468, the date neuron
PLANS = {
    "count": [("baseline", [], 0.0, SIX, 10, 120)]
    + [("L14N8949", [(*MAIN, d)], d, SIX, 10, 120) for d in (0.5, 1.0)]
    + [(f"random-L14N{n}", [(14, n, 1.0)], 1.0, SIX, 2, 120) for n in RANDOM_L14],
    "ladder": [("L14N8949", [(*MAIN, d)] if d else [], d, ["ko-promo"], 6, 160) for d in (0.0, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0)],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="count,ladder")
    ap.add_argument("--quant", default="4bit", choices=["4bit", "8bit", "bf16"])
    ap.add_argument("--batch", type=int, default=30)
    ap.add_argument("--seed", type=int, default=4040)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    tok, model, layers, stop = load(a.quant)
    plans = [p for p in a.plan.split(",") if p]
    pairs = sorted({(L, n) for p in plans for _, recipe, *_ in PLANS[p] for L, n, _ in recipe})
    cols, medians = calibrate(model, layers, tok, pairs)
    seed = a.seed
    for plan in plans:
        records = []
        for name, recipe, scale, keys, samples, max_new in PLANS[plan]:
            items = [(k, s) for k in keys for s in range(samples)]
            hooks, info = steer(layers, recipe, medians, cols)
            t = time.time()
            try:
                for i in range(0, len(items), a.batch):
                    chunk = items[i:i + a.batch]
                    seed += 1
                    outs = generate(model, tok, [P[k] for k, _ in chunk], seed, max_new, stop)
                    for (k, s), text in zip(chunk, outs):
                        records.append({"id": len(records), "plan": f"single-{plan}", "recipe": name, "scale": scale,
                                        "prompt_key": k, "prompt": P[k], "sample": s, "seed": seed, "text": text})
            finally:
                for h in hooks:
                    h.remove()
            log(f"{plan} {name} x{scale}: {len(items)} answers in {time.time() - t:.0f}s {info}")
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        meta = {"quant": a.quant, "plan": plan, "main": list(MAIN), "random_l14": RANDOM_L14,
                "sampling": "temperature 0.7, top_p 0.8, top_k 20, thinking off",
                "medians": {f"L{L}": v for L, v in medians.items()}, "minutes": round((time.time() - t0) / 60, 1)}
        out = HERE / "results" / f"{stamp}-single-{plan}.json"
        out.write_text(json.dumps({"meta": meta, "records": records}, ensure_ascii=False, indent=1), encoding="utf-8")
        log(f"wrote results/{out.name}: {len(records)} answers")


if __name__ == "__main__":
    main()
