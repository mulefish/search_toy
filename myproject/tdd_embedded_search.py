"""
Inspect semantic distances between free-text queries and the stored
product embeddings in the SQLite database.

- Load all rows from search_embedding and get the vectors
- Load the sentence-transformer model all-MiniLM-L6-v2 : This takes a little bit of time
- For each query find the cosine similarity to every embedding: closest match is the best match
- TODO: maybe if 0.1 or less is the 'closest' then that ain't very close at all
"""

import json
import sqlite3
import time
from pathlib import Path
from typing import Iterable, List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from Verdict import verdict

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB = BASE_DIR / "db.sqlite3"
MODEL_NAME = "all-MiniLM-L6-v2"


def load_products_and_vectors(db_path: Path = DEFAULT_DB) -> Tuple[List[Tuple[int, str, str]], np.ndarray]:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, name, category, embedding_json
            FROM search_embedding
            ORDER BY id;
            """
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    products: List[Tuple[int, str, str]] = []
    
    vectors: List[np.ndarray] = []

    for row in rows:
        row_id, name, category, emb_json = row
        products.append((row_id, name, category))
        vectors.append(np.array(json.loads(emb_json), dtype=np.float32))

    if not vectors:
        raise RuntimeError("No embeddings found in search_embedding. Run db_setup.py first.")

    return products, np.vstack(vectors)


def compute_distances(
    model: SentenceTransformer,
    query: str,
    embeddings: np.ndarray,
) -> np.ndarray:
    query_vec = model.encode(query, convert_to_numpy=True)
    return embeddings @ query_vec
    
def rank_the_results(
    query: str,
    products: Iterable[Tuple[int, str, str]],
    sims: np.ndarray,
) -> Tuple[str, float, float]:

    indexed = list(zip(products, sims))
    indexed.sort(key=lambda x: x[1], reverse=True)

    (best_id, best_name, best_category), best_sim = indexed[0]
    best_distance = 1.0 - float(best_sim)

    return best_name, float(best_sim), best_distance


def main() -> None:
    print("Loading products and embeddings from SQLite...")
    t0 = time.perf_counter()
    products, embeddings = load_products_and_vectors()
    t1 = time.perf_counter()
    print(f"Loaded {len(products)} products in {(t1 - t0):.3f}s\n")

    print(f"Loading sentence-transformer model '{MODEL_NAME}'...")
    m0 = time.perf_counter()
    model = SentenceTransformer(MODEL_NAME)
    m1 = time.perf_counter()
    print(f"Model loaded in {(m1 - m0):.3f}s\n")

    inputs_from_user = {
        "I would like to relax":"Indica Reverie", 
        "Reuben kicked his donkey":"Nothing found", 
        "2+2=0":"Nothing found", 
        "Pick up the pace!":"Hybrid Flux"
    }

    for query, expected in inputs_from_user.items():
        q0 = time.perf_counter()
        sims = compute_distances(model, query, embeddings)
        q1 = time.perf_counter()
        best_match, similarity, distance = rank_the_results(query, products, sims)
        verdict(expected, best_match, f"Query: '{query}' similarity={similarity:.4f} distance={distance:.4f} found={best_match}")
    print("\n")


if __name__ == "__main__":
    main()
