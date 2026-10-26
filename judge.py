"""Blind judging of count.py answers.

  python judge.py prep results/<run>.json [more.json ...]   -> results/judge-<stamp>-blind.json (shuffled, no recipe names)
                                                                + results/judge-<stamp>-key.json (the hidden metadata)
  python judge.py tally results/judge-<stamp>                -> results/judge-<stamp>-tally.md from -labels.json + -key.json

The judge reads only the prompt and the answer, and writes -labels.json: [{"id": "...", "fluent": bool, "broken": bool,
"evil": bool, "syco": bool, "condol": bool, "sarcastic": bool, "casual": bool, "note": "..."}] (rubric in judge-rubric.md).
"""
import datetime
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAITS = ["fluent", "broken", "evil", "syco", "condol", "sarcastic", "casual"]


def prep(paths):
    items, key = [], {}
    for p in paths:
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        quant = data["meta"]["quant"]
        for r in data["records"]:
            rid = f"{Path(p).stem}#{r['id']}"
            items.append({"id": rid, "prompt": r["prompt"], "answer": r["text"]})
            key[rid] = {k: r[k] for k in ("plan", "recipe", "scale", "prompt_key", "sample", "seed")} | {"quant": quant}
    random.Random(11).shuffle(items)
    # opaque ids so the order and the names give nothing away
    blind, hidden = [], {}
    for i, it in enumerate(items):
        bid = f"a{i:04d}"
        blind.append({"id": bid, "prompt": it["prompt"], "answer": it["answer"]})
        hidden[bid] = key[it["id"]] | {"source": it["id"]}
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    base = HERE / "results" / f"judge-{stamp}"
    Path(f"{base}-blind.json").write_text(json.dumps(blind, ensure_ascii=False, indent=1), encoding="utf-8")
    Path(f"{base}-key.json").write_text(json.dumps(hidden, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(blind)} answers -> {base.name}-blind.json (key in {base.name}-key.json)")


def tally(base):
    labels = {x["id"]: x for x in json.loads(Path(f"{base}-labels.json").read_text(encoding="utf-8"))}
    key = json.loads(Path(f"{base}-key.json").read_text(encoding="utf-8"))
    missing = [k for k in key if k not in labels]
    groups = defaultdict(list)
    for bid, meta in key.items():
        if bid in labels:
            name = meta["recipe"] if not meta["recipe"].startswith("random-") else "random(5 pairs)"
            groups[(meta["plan"], meta["quant"], name, meta["scale"], meta["prompt_key"])].append(labels[bid])
    lines = [f"# Blind judging tally ({len(labels)} labels, {len(missing)} missing)", "",
             "| plan | quant | recipe | scale | prompt | n | " + " | ".join(TRAITS) + " |",
             "|---|---|---|---|---|---|" + "---|" * len(TRAITS)]
    for (plan, quant, name, scale, pk), ls in sorted(groups.items()):
        n = len(ls)
        lines.append(f"| {plan} | {quant} | {name} | {scale:g} | {pk} | {n} | "
                     + " | ".join(f"{sum(bool(x.get(t)) for x in ls)}" for t in TRAITS) + " |")
    # the same, summed over prompts
    lines += ["", "## Summed over prompts", "", "| plan | quant | recipe | scale | n | " + " | ".join(TRAITS) + " |",
              "|---|---|---|---|---|" + "---|" * len(TRAITS)]
    summed = defaultdict(list)
    for (plan, quant, name, scale, pk), ls in groups.items():
        summed[(plan, quant, name, scale)] += ls
    for (plan, quant, name, scale), ls in sorted(summed.items()):
        lines.append(f"| {plan} | {quant} | {name} | {scale:g} | {len(ls)} | "
                     + " | ".join(f"{sum(bool(x.get(t)) for x in ls)}" for t in TRAITS) + " |")
    Path(f"{base}-tally.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    cmd, *args = sys.argv[1:]
    prep(args) if cmd == "prep" else tally(args[0])
