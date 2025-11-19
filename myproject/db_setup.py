"""
Create or refresh the SQLite database with:
- A search_item table seeded with sample hype-text records.
- A search_embedding table that stores one embedding vector per item
"""

import argparse
import json
import sqlite3
from pathlib import Path

from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB = BASE_DIR / "db.sqlite3"

MODEL_NAME = "all-MiniLM-L6-v2"

ITEMS = [
    (
        "Indica Reverie",
        "Indica drapes the evening in velvet calm, melting every muscle into the couch while citrus-lavender terpenes lull your mind toward cinematic dreams and whispery conversations that feel like bonus tracks from a favorite album. The buzz lingers like a weighted blanket, turning every flicker of candlelight into a widescreen spectacle and inviting deep breaths of hush.",
        "Indica",
    ),
    (
        "Sativa Voltage",
        "Sativa fires up neon-focus swagger, a sunrise surge that sends your thoughts sprinting across skyline billboards with laser-sharp wit and unstoppable creative flow. Each inhale feels like cracking open a can of inspiration, launching brainstorming sessions, rooftop hangouts, and fearless to-do list takedowns with equal amounts of gleam.",
        "Sativa",
    ),
    (
        "Hybrid Flux",
        "Hybrid splices the genetics of hustle and hush, letting you toggle between brainstorm brilliance and after-hours bliss with a single smooth inhale. It's the Swiss Army buzz—boardroom presentable one minute, fire-pit stories the next—keeping your vibe cruising on adaptive cruise control no matter how the day rearranges itself.",
        "Hybrid",
    ),
    (
        "Wax Prism",
        "Wax condenses flavor galaxies into molten gold, snapping onto your rig with terpene fireworks that crackle like a headline festival drop. One pull paints your palate with citrus candy, pine forest, and backstage-pass confidence, leaving a shimmering trail of dense clouds that shout premium without saying a word.",
        "Wax",
    ),
    (
        "Pre-Roll Ember",
        "Pre-rolls roll out concierge convenience, artisan-packed cones that spark evenly and unfurl luxury smoke trails wherever the night takes you. Slide a tube into your pocket and you're packing a mobile lounge—perfectly ground flower, slow-burning papers, and a vibe that says you paid attention to the finer details.",
        "Pre-roll",
    ),
    (
        "Gummy Aurora",
        "Gummys glow with technicolor delight, micro-dosed jewels that melt beneath the tongue and glide you through a syrupy spectrum of euphoria. Each bite is a postcard from Candyland—precision-dosed, terpene-kissed, and ready to turn playlists, painting sessions, or spa-night rituals into a silky-smooth adventure.",
        "Gummys",
    ),
    (
        "Pre-Rolls Duo",
        "Pre-rolls pair up in twin packs for co-op adventures, sharing boutique flower that burns clean, smells like a terpene boutique, and keeps the vibe synchronized from first spark to final ash. Whether you're matching cones with a friend or saving one for later, each wrap feels like a pocket-sized house party.",
        "Pre-rolls",
    ),
]

ITEMS_DETAILS = [
    ("Indica Reverie", 1, "{'something': 'something else'}"),
    ("Sativa Voltage", 2, "{'something': 'something else'}"),
    ("Hybrid Flux", 3, "{'something': 'something else'}"),
    ("Wax Prism", 4, "{'something': 'something else'}"),
    ("Pre-Roll Ember", 50, "{'something': 'something else'}"),
    ("Gummy Aurora", 66, "{'something': 'something else'}"),
    ("Pre-Rolls Duo", 77.77, "{'something': 'something else'}"),
]


def init_db(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    try:
        with conn:
            conn.execute("DROP TABLE IF EXISTS search_item;")
            conn.execute(
                """
                CREATE TABLE search_item (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name VARCHAR(120) NOT NULL UNIQUE,
                    description TEXT NOT NULL,
                    category VARCHAR(80) NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.executemany(
                "INSERT INTO search_item (name, description, category) VALUES (?, ?, ?);",
                ITEMS,
            )
            print("search_item table created and populated")

            conn.execute("DROP TABLE IF EXISTS search_embedding;")
            conn.execute(
                """
                CREATE TABLE search_embedding (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    FOREIGN KEY (item_id) REFERENCES search_item(id) ON DELETE CASCADE
                );
                """
            )
            print("search_embedding table created")

            conn.execute("DROP TABLE IF EXISTS search_item_details;")
            conn.execute(
                """
                CREATE TABLE search_item_details (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    number REAL NOT NULL,
                    json_string TEXT NOT NULL
                );
                """
            )
            conn.executemany(
                "INSERT INTO search_item_details (name, number, json_string) VALUES (?, ?, ?);",
                ITEMS_DETAILS,
            )
            print("search_item_details table created and populated")

        model = SentenceTransformer(MODEL_NAME)
        texts = [f"{name}. {description}" for name, description, _ in ITEMS]
        embeddings = model.encode(texts, convert_to_numpy=True)

        with conn:
            # Fetch item ids in insertion order
            cursor = conn.execute(
                "SELECT id, name, category FROM search_item ORDER BY id;"
            )
            item_rows = cursor.fetchall()

            rows = []
            for (item_id, name, category), vector in zip(item_rows, embeddings):
                rows.append(
                    (
                        item_id,
                        name,
                        category,
                        MODEL_NAME,
                        json.dumps(vector.tolist()),
                    )
                )

            conn.executemany(
                "INSERT INTO search_embedding (item_id, name, category, model_name, embedding_json) VALUES (?, ?, ?, ?, ?);",
                rows,
            )
            print(f"search_embedding table created and populated ({len(rows)} rows)")
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create/seed the SQLite database with items and embeddings."
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help=f"Path to DB (default: {DEFAULT_DB})")
    args = parser.parse_args()

    args.db.parent.mkdir(parents=True, exist_ok=True)
    init_db(args.db)


if __name__ == "__main__":
    main()




