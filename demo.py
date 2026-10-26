"""Steer one neuron of Qwen3.5-4B and stream the answer.

  python demo.py "오늘 회사에서 승진했어!"                               # L14 N8949 at strength 0.5 (the 'villain' neuron)
  python demo.py --strength 0 "오늘 회사에서 승진했어!"                  # no steering, for comparison
  python demo.py --layer 3 --neuron 5161 --strength -0.8 "Hi! Who are you?"   # a neuron someone else shared

Strength 1 adds the layer's median residual-stream norm along the neuron's output direction (see steer.py); strength
2 and above usually breaks the model's language. Layers and neurons count from 0.
"""
import argparse
import sys
import threading

import torch
from transformers import TextIteratorStreamer

from steer import calibrate, chat, load, steer


def main():
    ap = argparse.ArgumentParser(description="Steer one neuron of Qwen3.5-4B and stream its answer.")
    ap.add_argument("prompt", nargs="?", default="오늘 회사에서 승진했어!")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--neuron", type=int, default=8949)
    ap.add_argument("--strength", type=float, default=0.5)
    ap.add_argument("--quant", default="4bit", choices=["4bit", "8bit", "bf16"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=160)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    tok, model, layers, stop = load(a.quant)
    cols, medians = calibrate(model, layers, tok, [(a.layer, a.neuron)])
    hooks, info = steer(layers, [(a.layer, a.neuron, a.strength)], medians, cols) if a.strength else ([], [])
    print(f"\nL{a.layer} N{a.neuron} strength {a.strength:g} {info}\n\nyou > {a.prompt}\n")
    try:
        enc = tok(chat(tok, a.prompt), return_tensors="pt").to(model.device)
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        torch.manual_seed(a.seed)
        kw = dict(**enc, max_new_tokens=a.max_new, do_sample=True, temperature=0.7, top_p=0.8, top_k=20,
                  eos_token_id=stop, pad_token_id=tok.pad_token_id, streamer=streamer)
        th = threading.Thread(target=model.generate, kwargs=kw)
        th.start()
        print("AI  > ", end="", flush=True)
        for piece in streamer:
            print(piece, end="", flush=True)
        th.join()
        print()
    finally:
        for h in hooks:
            h.remove()


if __name__ == "__main__":
    main()
