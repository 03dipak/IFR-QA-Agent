import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=PROJECT_ROOT / "results/full_index_v2/bge_small")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    index_dir = args.index if args.index.is_absolute() else PROJECT_ROOT / args.index
    out_dir = args.out or index_dir
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ids_path = index_dir / "chunk_ids.json"
    vectors_path = index_dir / "embeddings.npy"
    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    vectors = np.load(vectors_path, mmap_mode="r")
    if vectors.shape[0] != len(ids):
        raise SystemExit("embeddings.npy rows disagree with chunk_ids.json")

    target = out_dir / "vectors.npy"
    np.save(target, np.asarray(vectors, dtype=np.float32))
    metadata = {
        "chunk_ids": ids,
        "rows": vectors.shape[0],
        "dim": int(vectors.shape[1]),
        "source": str(vectors_path),
        "source_sha256": _sha256_file(vectors_path),
    }
    (out_dir / "vector_meta.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {target} and {out_dir / 'vector_meta.json'} rows={vectors.shape[0]} dim={vectors.shape[1]}")


if __name__ == "__main__":
    main()
