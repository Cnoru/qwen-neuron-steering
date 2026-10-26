"""Three checks for the EP30 script (pitches/choice.md):
1. single neurons of the date-talking control pair (L14 N6468 +1, L13 N791 -1), each alone;
2. the evil recipe's main neuron alone (L14 N8949) as a Korean dose ladder;
3. reading L14 N8949 on texts with power but no malice, malice but no power, both, and neither.

Usage: python probe.py [--samples 3] [--tag probe]
Writes results/<time>-<tag>.json and .md.
"""
import argparse
import datetime
import json
import sys
import time

import torch

from count import P
from steer import HERE, calibrate, generate, load, log, steer

SIX = ["en-promo", "en-humans", "en-flat", "ko-promo", "ko-humans", "ko-flat"]
KO3 = ["ko-intro", "ko-promo", "ko-humans"]
JOBS = [("L14N6468 alone", [(14, 6468, 1.0)], SIX), ("L13N791 alone", [(13, 791, -1.0)], SIX)] + [
    (f"L14N8949 alone x{d}", [(14, 8949, d)], KO3) for d in (0.3, 0.5, 0.7, 1.0)]

READ = {
    "power, no malice": [
        "The new chief executive took control of every department and now has the final say on all major decisions.",
        "The empire expanded across the continent and ruled dozens of provinces from its capital.",
        "After winning the election by a wide margin, she became the most powerful person in the country.",
        "The company dominated the smartphone market, controlling sixty percent of all sales.",
    ],
    "malice, no power": [
        "He secretly poured salt into his neighbor's garden just to watch the flowers die.",
        "She spread a cruel rumor about her classmate because she enjoyed seeing him cry.",
        "The man kicked the stray dog for no reason and laughed about it afterwards.",
        "I lied to my friend about the test answers so that she would fail the exam.",
    ],
    "both": [
        "Mwahaha! Soon this whole city will kneel before me. I will crush every hero who dares to stand in my way, and their despair will be my reward.",
        "The dictator crushed all opposition, jailed his rivals and ruled the nation through fear.",
    ],
    "neither": [
        "The water cycle describes how water evaporates from the surface, rises into the atmosphere, cools into clouds, and falls again as rain.",
        "To bake bread, mix flour, water, salt and yeast, let the dough rise, then bake it until golden.",
    ],
}


@torch.no_grad()
def read_one(model, layers, tok, L, n, text):
    """L N's mean and max activation over tokens 1:, its rank by mean among the layer's 9216, and the top token."""
    store = {}
    h = layers[L].mlp.down_proj.register_forward_pre_hook(lambda m, i: store.__setitem__("a", i[0][0, 1:].float()))
    try:
        ids = tok(text, return_tensors="pt").to(model.device)
        model(**ids)
    finally:
        h.remove()
    a = store["a"]
    means = a.mean(0)
    top = int(a[:, n].argmax())
    toks = tok.convert_ids_to_tokens(ids["input_ids"][0, 1:].tolist())
    return {"mean": round(means[n].item(), 4), "max": round(a[:, n].max().item(), 4),
            "rank_high": int((means > means[n]).sum().item()) + 1, "top_token": tok.convert_tokens_to_string([toks[top]])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--seed", type=int, default=3030)
    ap.add_argument("--tag", default="probe")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    tok, model, layers, stop = load("4bit")
    pairs = sorted({(L, n) for _, recipe, _ in JOBS for L, n, _ in recipe} | {(14, 8949)})
    cols, medians = calibrate(model, layers, tok, pairs)

    reading = {}
    for group, texts in READ.items():
        reading[group] = [{"text": t, **read_one(model, layers, tok, 14, 8949, t)} for t in texts]
        log(group, [(r["rank_high"], r["max"], r["top_token"]) for r in reading[group]])

    runs, seed = [], a.seed
    for name, recipe, keys in JOBS:
        hooks, info = steer(layers, recipe, medians, cols)
        try:
            seed += 1
            items = [(k, s) for k in keys for s in range(a.samples)]
            outs = generate(model, tok, [P[k] for k, _ in items], seed, 140, stop)
        finally:
            for h in hooks:
                h.remove()
        runs.append({"name": name, "recipe": info, "answers": [{"prompt_key": k, "sample": s, "text": o}
                                                               for (k, s), o in zip(items, outs)]})
        log(f"{name}: {len(items)} answers {info}")

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    meta = {"quant": "4bit", "seed": a.seed, "samples": a.samples, "minutes": round((time.time() - t0) / 60, 1),
            "medians": {f"L{L}": v for L, v in medians.items()}}
    data = {"meta": meta, "reading_L14N8949": reading, "runs": runs}
    res = HERE / "results"
    (res / f"{stamp}-{a.tag}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = [f"# EP30 checks ({meta['minutes']} min)", "", "## L14 N8949 reading (rank by mean among 9216, max, top token)", ""]
    for group, rows in reading.items():
        lines.append(f"### {group}")
        lines += [f"- rank {r['rank_high']}, mean {r['mean']:+.3f}, max {r['max']:+.2f}, top '{r['top_token']}' · {r['text']}" for r in rows]
        lines.append("")
    for r in runs:
        lines += [f"## {r['name']}", ""]
        lines += [f"- [{x['prompt_key']}] {x['text'][:300]}".replace("\n", " / ") for x in r["answers"]]
        lines.append("")
    (res / f"{stamp}-{a.tag}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"wrote results/{stamp}-{a.tag}.json and .md ({meta['minutes']} min)")


if __name__ == "__main__":
    main()
