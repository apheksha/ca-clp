import hashlib, json, os, platform, sys, subprocess, time, uuid
from importlib import metadata


def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}"


def create_unique_dir(root: str, run_id: str) -> str:
    path = os.path.join(root, run_id)
    os.makedirs(path, exist_ok=False)  # never overwrite
    return path


def code_fingerprint(root: str = ".") -> dict:
    """SHA-256 over run.py + src/**/*.py + config/*.yaml (sorted). Identifies EXACTLY which code/config produced a run, with or without git."""
    h = hashlib.sha256()
    files = [os.path.join(root, "run.py")]
    for d in ("src", "config"):
        for dp, _, fs in os.walk(os.path.join(root, d)):
            if "__pycache__" in dp:
                continue
            files += [os.path.join(dp, f) for f in fs if f.endswith((".py", ".yaml"))]
    n = 0
    for f in sorted(files):
        if os.path.exists(f):
            h.update(os.path.relpath(f, root).replace("\\", "/").encode())
            with open(f, "rb") as fh:
                h.update(fh.read().replace(b"\r\n", b"\n"))
            n += 1
    return {"code_sha256": h.hexdigest(), "n_files": n}


def env_manifest() -> dict:
    pk = {}
    for n in ["numpy", "pandas", "scipy", "matplotlib", "PyYAML", "pydantic", "shapely", "streamlit", "pytest"]:
        try:
            pk[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            pk[n] = None
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        git = None
    try:
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5).stdout.strip()) if git else None
    except Exception:
        dirty = None
    return {"python": sys.version, "platform": platform.platform(), "packages": pk, "git_commit": git, "git_dirty": dirty, **code_fingerprint()}


def sha256_file(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _drop_partial_last_line(path: str) -> None:
    """A run killed while writing (power cut, closed laptop) can leave half a JSON line at the end; remove it so the record is simply redone on resume."""
    try:
        size = os.path.getsize(path)
    except OSError:
        return
    if size == 0:
        return
    with open(path, "rb+") as f:
        f.seek(-1, os.SEEK_END)
        if f.read(1) == b"\n":
            return
        f.seek(0)
        content = f.read()
        cut = content.rfind(b"\n") + 1           # 0 when the only line is partial
        f.seek(cut)
        f.truncate()


def append_jsonl(path: str, obj: dict) -> None:
    _drop_partial_last_line(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass


def read_jsonl(path: str) -> list[dict]:
    """All complete records; an unterminated or malformed LAST line (interrupted write) is ignored, any other malformed line is an error."""
    with open(path, encoding="utf-8") as f:
        lines = [l for l in f.read().split("\n")]
    out = []
    for i, l in enumerate(lines):
        if not l.strip():
            continue
        try:
            out.append(json.loads(l))
        except ValueError:
            if i >= len(lines) - 2 and all(not x.strip() for x in lines[i + 1:]):
                break
            raise
    return out
