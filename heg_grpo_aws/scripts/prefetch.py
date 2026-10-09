"""Download models and datasets once during setup, so paid training time is never spent downloading."""
import argparse
import os
import sys
import time
from pathlib import Path

# The hf-xet download path has no stall timeout: in the WSL rehearsal (2026-10-04) it stopped the
# 988 MB Qwen weights at 134 MB for 45 min. Plain HTTP fetched the rest in 44 s and times out on
# stalls. Must be set before anything imports huggingface_hub.
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlvr.config import EMBED_MODEL, MODEL_NAME_BY_SIZE
from rlvr.data import MATH_DATASET, MATH_SUBJECT_CONFIGS


def retry(what, fn, attempts=4):
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # network errors surface as many different types
            if i == attempts - 1:
                raise
            print(f'  {what}: {type(e).__name__}: {e} -> retry {i + 2}/{attempts} in 15 s')
            time.sleep(15)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sizes', default='0.5B', help='comma list of model sizes, e.g. 0.5B,1.5B')
    args = p.parse_args()

    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer

    for size in args.sizes.split(','):
        name = MODEL_NAME_BY_SIZE[size]
        print(f'model {name}')
        retry(name, lambda: snapshot_download(
            name, allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', 'merges.txt']))
    print(f'embedder {EMBED_MODEL}')
    retry(EMBED_MODEL, lambda: SentenceTransformer(EMBED_MODEL, device='cpu'))
    n_math = 0
    for cfg in MATH_SUBJECT_CONFIGS:
        n_math += len(retry(cfg, lambda: load_dataset(MATH_DATASET, cfg, split='train')))
    print(f'dataset {MATH_DATASET}: {n_math} training problems across {len(MATH_SUBJECT_CONFIGS)} subjects')
    print('dataset HuggingFaceH4/MATH-500:',
          len(retry('MATH-500', lambda: load_dataset('HuggingFaceH4/MATH-500', split='test'))))
    for split in ('train', 'test'):
        print(f'dataset openai/gsm8k {split}:',
              len(retry('gsm8k', lambda: load_dataset('openai/gsm8k', 'main', split=split))))
    print('prefetch complete')


if __name__ == '__main__':
    main()
