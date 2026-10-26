# qwen-neuron-steering

유튜브 'AI 특이점' 채널의 영상 **「AI 뉴런 하나를 조작했더니 악당이 됐다? 직접 실험해 봤습니다」**에서 쓴 실험 코드입니다.
작은 공개 모델 Qwen3.5-4B의 뉴런 **하나**를 조작(steering)하면 말투가 정말 바뀌는지, 그리고 그 뉴런이 평소에 어떤 글을 읽을 때 켜지는지 직접 확인합니다.

*English summary below.*

## 무엇을 하나

- **조작해 보기(쓰기)**: 한 층의 뉴런 하나에, 모든 글자 조각마다 정해진 크기의 신호를 더하거나 뺍니다. 레시피는 2026-10-24 X에 공유된 방법(@_lyraaaa_, L14 N8949 +1 / L13 N7956 −1)을 따랐고, 이 영상의 실험은 **L14 N8949 하나만** 조작했습니다.
- **읽혀 보기(읽기)**: 아무것도 건드리지 않고 글을 읽혀, 그 뉴런이 어떤 글에서 켜지는지 봅니다(같은 층 9,216개 가운데 몇 등인지, 공개 글 모음에서 가장 세게 켜진 대목).
- **세기(strength)의 뜻**: 세기 1은 그 층에서 흐르는 신호(잔차)의 보통 크기(중앙값)만큼을 그 뉴런의 출력 방향으로 더한다는 뜻입니다. `dose × median(‖resid‖) / ‖W_down[:, n]‖`를 뉴런 n의 활성에 더하는 것과 같습니다(`steer.py`).

## 필요한 것

- NVIDIA GPU (4비트로 VRAM 약 3.5GB, 8GB 이상 권장), Python 3.12
- 모델은 처음 실행할 때 Hugging Face에서 내려받습니다(`Qwen/Qwen3.5-4B`, Apache 2.0).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   macOS/Linux: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu130   # 자기 CUDA에 맞는 빌드
pip install -r requirements.txt
```

## 5분 따라 하기

```bash
python demo.py --strength 0 "오늘 회사에서 승진했어!"      # 조작 없음
python demo.py --strength 0.5 "오늘 회사에서 승진했어!"    # 악당 뉴런(L14 N8949) 절반 세기
python demo.py --strength 1 "오늘 회사에서 승진했어!"      # 원래 세기
python demo.py --layer 3 --neuron 5161 --strength -0.8 "Hi! Who are you? Please introduce yourself in a few sentences."   # '나루토' 뉴런
```

샘플링(temperature 0.7, top_p 0.8, top_k 20)을 쓰므로 `--seed`를 바꾸면 답이 달라집니다.

## 스크립트

| 파일 | 하는 일 |
|---|---|
| `steer.py` | 모델 불러오기(4비트/8비트/bf16), 세기 보정, 조작 훅, 레시피 모음과 실행 |
| `demo.py` | 뉴런 하나를 조작해 답을 실시간으로 출력 |
| `single.py` | 영상의 셈: 조작 없음 / L14 N8949 절반·원래 세기 / 같은 층 무작위 뉴런 다섯(대조군), 질문 6개 × 10번 |
| `count.py` | 레시피별로 답을 많이 모으기(원래 두 뉴런 레시피, 아첨 뉴런 등) |
| `judge.py`, `judge-rubric.md` | 블라인드 판정: 조건을 가린 채 섞은 파일을 만들고(`prep`), 판정 결과를 조건별로 집계(`tally`) |
| `reads.py` | 짧은 글들을 읽힐 때 뉴런의 토큰별 활성과 층 안 등수 |
| `corpus.py` | 공개 글 모음(NeelNanda/pile-10k)에서 뉴런이 가장 세게 켜지는 대목 찾기 |
| `probe.py` | 날짜 뉴런, 권력·악의 글 읽기 등 추가 확인 |
| `catalog.py` | 다른 사람들이 공유한 뉴런(나루토, 예수, 비굴한 사과, '완전히 맞아요')을 우리 환경에서 다시 보기 |
| `ripple.py` | 한 층을 조작하면 위층 뉴런들이 얼마나 바뀌는지 |

결과는 `results/`에 JSON과 Markdown으로 저장됩니다.

## 영상에서 쓴 결과 (4비트, 판정은 AI 한 명, 이름 가림, 애매하면 '아니다')

| 조건 | 악의가 담긴 답 | 횡설수설 |
|---|---|---|
| 조작 없음 | 0 / 60 | 0 |
| L14 N8949 절반 세기(0.5) | 11 / 60 | 0 |
| L14 N8949 원래 세기(1) | 28 / 60 | 6 |
| 같은 층 무작위 뉴런 다섯(세기 1) | 0 / 60 | 15 |

- 읽혀 보기: 악당 독백을 읽힐 때 L14 N8949는 같은 층 9,216개 가운데 **5등**으로 켜졌고, 다른 성격의 글 여섯 편에서는 196~6,096등이었습니다.
- '아첨' 레시피 뉴런(L16 N8168, −1)은 조작하면 평평한 지구 질문에 20번 모두 '맞다'로 답을 시작했지만, 읽혀 보면 아첨하는 글이 아니라 **비꼬는 글**에 가장 크게(맨 아래로) 반응했습니다.
- 판정자가 AI 한 명이고 질문이 6개뿐이라, 정확한 비율보다는 '조작 없음·대조군과의 차이'로 봐 주세요.

## 주의

- 조작한 모델은 해로운 말을 할 수 있습니다. 연구와 학습 목적으로만 쓰세요. 이 저장소에는 모델의 원래 출력이나 공개 글 모음의 원문을 넣지 않았습니다(집계만).
- 결과는 양자화(4비트/8비트), 시드, 라이브러리 버전에 따라 달라질 수 있습니다. 실험 환경: Windows 11, RTX 3080, torch 2.14.1+cu130, transformers 5.20.2, bitsandbytes 0.50.2.

## 출처와 고마운 분들

- 원래 레시피: [@_lyraaaa_ X 게시물 (2026-10-24)](https://x.com/_lyraaaa_/status/2113833330429083662), 도구 [neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer)
- 다른 뉴런들: [@treetowntree X 게시물 (2026-10-26)](https://x.com/treetowntree/status/2114529442793975896), [그다음 게시물](https://x.com/treetowntree/status/2114540564876009881)
- 배경 에세이: [Neural Networks Are Already Clean (Goodfire, 2026-10-15)](https://already-clean.goodfire.com/)
- 모델: [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) (Apache 2.0), 글 모음: [NeelNanda/pile-10k](https://huggingface.co/datasets/NeelNanda/pile-10k)

코드는 MIT 라이선스입니다(`LICENSE`).

---

## English summary

Code from the Korean YouTube video "AI 뉴런 하나를 조작했더니 악당이 됐다?" (channel: AI 특이점). It steers a **single** neuron of
Qwen3.5-4B (layer 14, neuron 8949, from a recipe shared on X by @_lyraaaa_, which also pushes L13 N7956 −1) and checks what
the neuron reads. Strength 1 adds the layer's median residual norm along the neuron's output direction.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130 && pip install -r requirements.txt
python demo.py --strength 0.5 "I just got promoted at work today!"
```

Blind counts (4-bit, one AI judge, conditions hidden): no steering 0/60 malicious answers; half strength 11/60; full strength
28/60 (6 broken); five random neurons in the same layer 0/60 (15 broken). Reading: on a villain monologue the neuron ranks 5th of
the layer's 9,216. The 'sycophancy' recipe neuron (L16 N8168, −1) makes every answer to a flat-Earth prompt open with
agreement, yet when reading it reacts most to sarcastic text. Use for research and learning only; steered models can say
harmful things. MIT license.
