"""Prompt clustering for CB-GRPO (Req 5, Req 43) and the shared sentence-embedding loader."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans

from .config import EMBED_MODEL

_SENTENCE_MODEL_CACHE: dict = {}


def get_sentence_model(model_name: str = EMBED_MODEL, device: str = 'cpu'):
    key = (model_name, device)
    if key not in _SENTENCE_MODEL_CACHE:
        from sentence_transformers import SentenceTransformer
        _SENTENCE_MODEL_CACHE[key] = SentenceTransformer(model_name, device=device)
    return _SENTENCE_MODEL_CACHE[key]


def embed_texts(texts: list, model_name: str = EMBED_MODEL, device: str = 'cpu',
                batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
    model = get_sentence_model(model_name, device)
    emb = model.encode(list(texts), batch_size=batch_size, show_progress_bar=show_progress,
                       convert_to_numpy=True, normalize_embeddings=True)
    return emb.astype(np.float32)


def cluster_embeddings(embeddings: np.ndarray, n_clusters: int, basis: str = 'semantic',
                       solve_rates: Optional[np.ndarray] = None, difficulty_weight: float = 1.0,
                       random_state: int = 0) -> np.ndarray:
    """basis='semantic': KMeans on embeddings. basis='difficulty_aware' (Req 43): also on solve rate."""
    if basis == 'semantic':
        features = embeddings
    elif basis == 'difficulty_aware':
        if solve_rates is None:
            raise ValueError('difficulty_aware clustering requires per-prompt solve_rates (Req 43.1).')
        sr = np.asarray(solve_rates, dtype=np.float64).reshape(-1, 1)
        sr = (sr - sr.mean()) / (sr.std() + 1e-8)
        features = np.concatenate([embeddings, difficulty_weight * sr], axis=1)
    else:
        raise ValueError(f'Unknown clustering basis: {basis}')
    if len(features) < n_clusters:
        raise ValueError(f'n_clusters={n_clusters} exceeds number of prompts={len(features)}.')
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    return km.fit_predict(features).astype(np.int64)


def cluster_size_distribution(cluster_ids: np.ndarray, n_clusters: int) -> dict:
    counts = np.bincount(cluster_ids, minlength=n_clusters)
    return {int(c): int(n) for c, n in enumerate(counts)}


def save_cluster_assignments(cluster_ids: np.ndarray, prompt_ids: list, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({'prompt_ids': prompt_ids, 'cluster_ids': cluster_ids.tolist()}))


def load_cluster_assignments(path: str) -> tuple:
    payload = json.loads(Path(path).read_text())
    return payload['prompt_ids'], np.asarray(payload['cluster_ids'], dtype=np.int64)
