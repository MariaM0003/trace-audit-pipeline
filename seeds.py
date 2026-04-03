"""
TraceAudit Seed Vault
---------------------
20 clean, diverse Python functions spanning:
  - File I/O & Resource Management
  - API / HTTP Clients
  - Data Processing & Transformation
  - Concurrency & Threading
  - Authentication & Security

These are the "Healthy Patients" — correct implementations that the
Mutation Engine will corrupt with subtle semantic bugs.
"""

# ──────────────────────────────────────────────
# CATEGORY 1: File I/O & Resource Management
# ──────────────────────────────────────────────

def read_config(filepath: str) -> dict:
    """Read a JSON config file and return its contents as a dict."""
    import json
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def write_report(data: list[dict], output_path: str) -> None:
    """Write a list of dicts as newline-delimited JSON to a file."""
    import json
    with open(output_path, "w", encoding="utf-8") as f:
        for record in data:
            f.write(json.dumps(record) + "\n")


def copy_file_chunked(src: str, dst: str, chunk_size: int = 8192) -> int:
    """Copy a file in chunks; returns total bytes written."""
    total = 0
    with open(src, "rb") as source, open(dst, "wb") as dest:
        while chunk := source.read(chunk_size):
            dest.write(chunk)
            total += len(chunk)
    return total


def tail_file(filepath: str, n: int = 10) -> list[str]:
    """Return the last n lines of a text file."""
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return lines[-n:]


def merge_csv_files(paths: list[str], output_path: str) -> int:
    """Merge multiple CSV files (same schema) into one; returns row count."""
    import csv
    rows_written = 0
    header_written = False
    with open(output_path, "w", newline="", encoding="utf-8") as outfile:
        writer = None
        for path in paths:
            with open(path, "r", newline="", encoding="utf-8") as infile:
                reader = csv.DictReader(infile)
                if writer is None:
                    writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
                if not header_written:
                    writer.writeheader()
                    header_written = True
                for row in reader:
                    writer.writerow(row)
                    rows_written += 1
    return rows_written


# ──────────────────────────────────────────────
# CATEGORY 2: API / HTTP Clients
# ──────────────────────────────────────────────

def fetch_json(url: str, timeout: int = 10) -> dict:
    """Perform a GET request and return parsed JSON."""
    import urllib.request
    import json
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def paginated_fetch(base_url: str, max_pages: int = 5) -> list[dict]:
    """Fetch paginated API results up to max_pages; stops on empty page."""
    import urllib.request
    import json
    results = []
    for page in range(1, max_pages + 1):
        url = f"{base_url}?page={page}"
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if not data:
            break
        results.extend(data)
    return results


def post_json(url: str, payload: dict, token: str) -> dict:
    """POST JSON payload with Bearer token; returns response dict."""
    import urllib.request
    import json
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ──────────────────────────────────────────────
# CATEGORY 3: Data Processing & Transformation
# ──────────────────────────────────────────────

def normalize_scores(scores: list[float]) -> list[float]:
    """Min-max normalize a list of floats to [0, 1]."""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if lo == hi:
        return [0.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def batch_records(records: list, batch_size: int) -> list[list]:
    """Split a list into fixed-size batches (last batch may be smaller)."""
    return [records[i: i + batch_size] for i in range(0, len(records), batch_size)]


def deduplicate_by_key(records: list[dict], key: str) -> list[dict]:
    """Return records with duplicate key values removed (first occurrence kept)."""
    seen = set()
    result = []
    for record in records:
        val = record.get(key)
        if val not in seen:
            seen.add(val)
            result.append(record)
    return result


def rolling_average(values: list[float], window: int) -> list[float]:
    """Compute rolling mean with given window size; first (window-1) values use partial windows."""
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        window_vals = values[start: i + 1]
        result.append(sum(window_vals) / len(window_vals))
    return result


def flatten_nested(nested: list, depth: int = 1) -> list:
    """Flatten a nested list up to `depth` levels."""
    if depth == 0:
        return nested
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(flatten_nested(item, depth - 1))
        else:
            result.append(item)
    return result


# ──────────────────────────────────────────────
# CATEGORY 4: Concurrency & Threading
# ──────────────────────────────────────────────

def parallel_map(fn, items: list, max_workers: int = 4) -> list:
    """Apply fn to each item using a thread pool; preserves order."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(fn, items))


def rate_limited_worker(tasks: list, delay: float = 0.1) -> list:
    """Process tasks sequentially with a fixed delay between each."""
    import time
    results = []
    for task in tasks:
        results.append(task())
        time.sleep(delay)
    return results


def producer_consumer(items: list, process_fn, num_consumers: int = 2) -> list:
    """Classic producer-consumer using a Queue; thread-safe result collection."""
    import queue
    import threading

    q: queue.Queue = queue.Queue()
    results = []
    lock = threading.Lock()

    def consumer():
        while True:
            item = q.get()
            if item is None:
                q.task_done()
                break
            result = process_fn(item)
            with lock:
                results.append(result)
            q.task_done()

    threads = [threading.Thread(target=consumer) for _ in range(num_consumers)]
    for t in threads:
        t.start()

    for item in items:
        q.put(item)

    for _ in range(num_consumers):
        q.put(None)  # sentinel

    q.join()
    for t in threads:
        t.join()

    return results


# ──────────────────────────────────────────────
# CATEGORY 5: Auth & Security Utilities
# ──────────────────────────────────────────────

def hash_password(password: str, salt: bytes | None = None) -> tuple[str, str]:
    """Hash a password with PBKDF2-HMAC-SHA256; returns (hex_hash, hex_salt)."""
    import hashlib
    import os
    if salt is None:
        salt = os.urandom(32)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
    return key.hex(), salt.hex()


def verify_password(password: str, stored_hash: str, stored_salt: str) -> bool:
    """Re-derive the hash and compare to stored value using constant-time compare."""
    import hashlib
    import hmac
    salt = bytes.fromhex(stored_salt)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
    return hmac.compare_digest(key.hex(), stored_hash)


def generate_token(user_id: int, secret: str, ttl_seconds: int = 3600) -> str:
    """Create a simple signed token: base64(payload).HMAC-SHA256 signature."""
    import base64
    import hashlib
    import hmac
    import json
    import time

    payload = json.dumps({"uid": user_id, "exp": int(time.time()) + ttl_seconds})
    b64_payload = base64.urlsafe_b64encode(payload.encode()).decode()
    sig = hmac.new(secret.encode(), b64_payload.encode(), hashlib.sha256).hexdigest()
    return f"{b64_payload}.{sig}"


def validate_token(token: str, secret: str) -> dict | None:
    """Validate a signed token; returns payload dict or None if invalid/expired."""
    import base64
    import hashlib
    import hmac
    import json
    import time

    try:
        b64_payload, sig = token.split(".")
        expected_sig = hmac.new(secret.encode(), b64_payload.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected_sig):
            return None
        payload = json.loads(base64.urlsafe_b64decode(b64_payload + "==").decode())
        if payload["exp"] < int(time.time()):
            return None
        return payload
    except Exception:
        return None


# ──────────────────────────────────────────────
# Registry: used by generator.py
# ──────────────────────────────────────────────
import inspect

SEED_FUNCTIONS: list[dict] = []

_all_funcs = [
    read_config, write_report, copy_file_chunked, tail_file, merge_csv_files,
    fetch_json, paginated_fetch, post_json,
    normalize_scores, batch_records, deduplicate_by_key, rolling_average, flatten_nested,
    parallel_map, rate_limited_worker, producer_consumer,
    hash_password, verify_password, generate_token, validate_token,
]

for _fn in _all_funcs:
    SEED_FUNCTIONS.append({
        "name": _fn.__name__,
        "source": inspect.getsource(_fn),
        "docstring": (_fn.__doc__ or "").strip(),
    })
