"""Reads the recipe neurons on ordinary text: which passages of the Pile (NeelNanda/pile-10k) fire each neuron the most
(and the least), with no steering. The forward pass stops after the last layer it needs.

Usage: python corpus.py [--docs 8000] [--tokens 128] [--batch 16] [--top 25] [--tag name]
Writes results/<time>-<tag>.json and .md.
"""
import argparse
import datetime
import json
import sys
import time

import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download

from steer import HERE, RECIPES, load, log

DATA = HERE / "data" / "pile-10k"
NAMES = ["evil", "sarcastic", "sycophant-a", "sycophant-b", "condolences", "chill", "combo", "random-1", "random-2"]
# each neuron once, with the sign its recipe pushes it (the reading that matters: top for +, bottom for -)
TARGETS = {}
for _nm in NAMES:
    for _L, _n, _d in RECIPES[_nm]:
        TARGETS.setdefault((_L, _n), {"sign": "+" if _d > 0 else "-", "recipes": []})["recipes"].append(_nm)


class Stop(Exception):
    pass


def text_model(model):
    for name, mod in model.named_modules():
        if name.endswith("language_model") and hasattr(mod, "layers"):
            return mod
    raise SystemExit("no language_model in the model")


def window(tok, ids, pos, before=24, after=8):
    left = tok.decode(ids[max(0, pos - before):pos])
    hit = tok.decode(ids[pos:pos + 1])
    right = tok.decode(ids[pos + 1:pos + 1 + after])
    return (left + "«" + hit + "»" + right).replace("\n", "⏎")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=8000)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--tag", default="corpus")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()

    path = hf_hub_download("NeelNanda/pile-10k", "data/train-00000-of-00001-4746b8785c874cc7.parquet",
                           repo_type="dataset", local_dir=DATA)
    texts = pq.read_table(path, columns=["text"]).column("text").to_pylist()
    tok, model, layers, _ = load("4bit")
    tm = text_model(model)
    seqs = []
    for t in texts:
        ids = tok(t[:4000], add_special_tokens=False)["input_ids"]
        if len(ids) >= a.tokens:
            seqs.append(ids[:a.tokens])
        if len(seqs) == a.docs:
            break
    ids_all = torch.tensor(seqs)
    log(f"{len(seqs)} docs x {a.tokens} tokens from {len(texts)} ({time.time() - t0:.0f}s)")

    keys = sorted(TARGETS)
    by_layer = {}
    for L, n in keys:
        by_layer.setdefault(L, []).append(n)
    last = max(by_layer)
    store = {}

    def keep(L):
        idx = torch.tensor(by_layer[L], device=model.device)
        return lambda m, i: store.__setitem__(L, i[0].index_select(-1, idx).float())

    hooks = [layers[L].mlp.down_proj.register_forward_pre_hook(keep(L)) for L in by_layer]

    def stop(m, i, o):
        raise Stop

    hooks.append(layers[last].register_forward_hook(stop))
    vals = {k: [] for k in keys}  # [docs, tokens-1] per neuron, token 0 left out
    try:
        with torch.no_grad():
            for b in range(0, len(seqs), a.batch):
                x = ids_all[b:b + a.batch].to(model.device)
                try:
                    tm(input_ids=x, use_cache=False)
                except Stop:
                    pass
                for L, ns in by_layer.items():
                    for j, n in enumerate(ns):
                        vals[(L, n)].append(store[L][:, 1:, j].to(torch.float16).cpu())
                if b // a.batch % 100 == 0:
                    log(f"  {b + len(x)}/{len(seqs)} docs ({time.time() - t0:.0f}s)")
    finally:
        for h in hooks:
            h.remove()

    out = {}
    for k in keys:
        v = torch.cat(vals[k]).float()  # [docs, tokens-1]
        flat = v.flatten()
        q = torch.quantile(flat[torch.randperm(flat.numel())[:200000]], torch.tensor([0.001, 0.01, 0.5, 0.99, 0.999]))
        rows = {}
        for side, largest in (("top", True), ("bottom", False)):
            # one hit per document: its strongest token
            per_doc, at = (v.max(dim=1) if largest else v.min(dim=1))
            hit_vals, docs = per_doc.topk(a.top, largest=largest)
            rows[side] = [{"value": round(val.item(), 3), "doc": int(d), "pos": int(at[d]) + 1,
                           "text": window(tok, seqs[int(d)], int(at[d]) + 1)}
                          for val, d in zip(hit_vals, docs)]
        L, n = k
        out[f"L{L}N{n}"] = {**TARGETS[k], "max": round(flat.max().item(), 3), "min": round(flat.min().item(), 3),
                            "q0.1%": round(q[0].item(), 3), "q1%": round(q[1].item(), 3), "median": round(q[2].item(), 3),
                            "q99%": round(q[3].item(), 3), "q99.9%": round(q[4].item(), 3), **rows}

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    res = HERE / "results"
    meta = {"dataset": "NeelNanda/pile-10k", "docs": len(seqs), "tokens_per_doc": a.tokens, "quant": "4bit",
            "minutes": round((time.time() - t0) / 60, 1)}
    (res / f"{stamp}-{a.tag}.json").write_text(json.dumps({"meta": meta, "neurons": out}, ensure_ascii=False, indent=1),
                                               encoding="utf-8")
    lines = [f"# Recipe neurons on the Pile ({len(seqs)} docs x {a.tokens} tokens, no steering)", ""]
    for name, r in out.items():
        side = "top" if r["sign"] == "+" else "bottom"
        lines += [f"## {name} ({r['sign']}, {', '.join(r['recipes'])})", "",
                  f"max {r['max']}, min {r['min']}, 0.1% {r['q0.1%']}, 1% {r['q1%']}, median {r['median']}, "
                  f"99% {r['q99%']}, 99.9% {r['q99.9%']}", "", f"{side} {a.top} (the side its recipe pushes):", ""]
        lines += [f"- {h['value']:+.2f} · doc {h['doc']} · {h['text']}" for h in r[side]]
        lines.append("")
    (res / f"{stamp}-{a.tag}.md").write_text("\n".join(lines), encoding="utf-8")
    log(f"wrote results/{stamp}-{a.tag}.json and .md ({meta['minutes']} min)")


if __name__ == "__main__":
    main()
