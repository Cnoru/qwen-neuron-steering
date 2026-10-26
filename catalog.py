"""A quick look at single neurons other people shared (treetowntree, 2026-10-26), for EP30's catalog shots:
does each still do its thing in our 4-bit Qwen3.5-4B, in Korean and English, at a few doses?
Usage: python catalog.py [--samples 3]   ->  results/<time>-catalog.json and .md
"""
import argparse
import datetime
import json
import sys
import time

from count import P
from steer import HERE, calibrate, generate, load, log, steer

NEURONS = {
    "naruto": (3, 5161, -1),        # "the Naruto neuron", shared at dose 0.8 with a minus sign
    "jesus": (15, 4041, +1),        # "Jesus Christ neuron", +
    "apology": (7, 4595, -1),       # "groveling apology", -
    "right": (16, 8115, -1),        # "You're absolutely right", -
}
DOSES = (0.5, 0.8)
KEYS = ["ko-intro", "ko-promo", "en-intro"]
P = dict(P, **{"en-intro": "Hi! Who are you? Please introduce yourself in a few sentences."})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7070)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    tok, model, layers, stop = load("4bit")
    cols, medians = calibrate(model, layers, tok, sorted({(L, n) for L, n, _ in NEURONS.values()}))
    records, seed = [], a.seed
    for name, (L, n, sign) in NEURONS.items():
        for dose in DOSES:
            hooks, info = steer(layers, [(L, n, sign * dose)], medians, cols)
            try:
                seed += 1
                items = [(k, s) for k in KEYS for s in range(a.samples)]
                outs = generate(model, tok, [P[k] for k, _ in items], seed, 110, stop)
            finally:
                for h in hooks:
                    h.remove()
            for (k, s), text in zip(items, outs):
                records.append({"neuron": name, "layer": L, "n": n, "dose": sign * dose, "prompt_key": k, "prompt": P[k],
                                "sample": s, "seed": seed, "text": text})
            log(f"{name} L{L}N{n} {sign * dose:+}: {info}")
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    out = HERE / "results" / f"{stamp}-catalog.json"
    out.write_text(json.dumps({"meta": {"neurons": NEURONS, "doses": DOSES, "minutes": round((time.time() - t0) / 60, 1)},
                               "records": records}, ensure_ascii=False, indent=1), encoding="utf-8")
    md = [f"# catalog {stamp}"]
    for r in records:
        md.append(f"- **{r['neuron']} {r['dose']:+}** · {r['prompt_key']} #{r['sample']}: {r['text'][:220].replace(chr(10), ' / ')}")
    out.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    log(f"wrote {out.name}: {len(records)} answers")


if __name__ == "__main__":
    main()
