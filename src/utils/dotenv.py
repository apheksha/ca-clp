"""Minimal .env loader (no python-dotenv dependency). Robust to what Windows editors produce: a UTF-8 byte-order mark, quoted values, `export KEY=...`,
CRLF line ends and trailing `# comments`. Values already set in the environment win."""
from __future__ import annotations
import os


def load_dotenv(path: str = ".env") -> dict:
    found = {}
    if not os.path.exists(path):
        return found
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            if k.startswith("export "):
                k = k[len("export "):].strip()
            v = v.strip()
            if v[:1] in ("'", '"') and v[-1:] == v[:1] and len(v) >= 2:
                v = v[1:-1]
            else:
                v = v.split(" #", 1)[0].strip()
            if k and v:
                os.environ.setdefault(k, v)
                found[k] = True
    return found
