import json
import sqlite3
import time
from pathlib import Path
from typing import Iterable, List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB = BASE_DIR / "db.sqlite3"
MODEL_NAME = "all-MiniLM-L6-v2"


def load_products_and_vectors(db_path: Path = DEFAULT_DB) -> Tuple[List[Tuple[int, str, str]], np.ndarray]:
    """
    Load items and their embedding vectors from the search_embedding table.

    Returns:
        products: list of (id, name, category)
        vectors:  N x D numpy array of embeddings
    """
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
    """
    Compute cosine similarity and distance from query to each embedding.

    Returns an array of shape (N,) with similarity scores.
    Distance is simply 1 - similarity.
    """
    query_vec = model.encode(query, convert_to_numpy=True)
    return embeddings @ query_vec
    
def print_ranked_results(
    query: str,
    products: Iterable[Tuple[int, str, str]],
    sims: np.ndarray,
) -> None:

    indexed = list(zip(products, sims))
    # Sort by best match first: highest similarity (lowest distance)
    indexed.sort(key=lambda x: x[1], reverse=True)

    # Best match as its own variable
    (best_id, best_name, best_category), best_sim = indexed[0]
    best_distance = 1.0 - float(best_sim)
    print(
        f"BEST: [{best_id:02d}] {best_name} ({best_category})  "
        f"similarity={best_sim:.4f}  distance={best_distance:.4f}"
    )

    # print("\nAll results:")
    # for (row_id, name, category), sim in indexed:
    #     distance = 1.0 - float(sim)
    #     print(f"[{row_id:02d}] {name} ({category})  similarity={sim:.4f}  distance={distance:.4f}")


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

    inputs_from_user = ["I would like to relax", "Reuben kicked his donkey", "2+2=0","Pick up the pace!"]

    for query in inputs_from_user:
        q0 = time.perf_counter()
        sims = compute_distances(model, query, embeddings)
        q1 = time.perf_counter()
        print(f"\nComputed similarities for '{query}' in {(q1 - q0):.4f}s")
        print_ranked_results(query, products, sims)
    print("\n")


if __name__ == "__main__":
    main()
