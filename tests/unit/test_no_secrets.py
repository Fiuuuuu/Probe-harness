from __future__ import annotations

import re
from pathlib import Path

FORBIDDEN = re.compile(
    r"(sk-[a-zA-Z0-9]{10,}|DEEPSEEK_API_KEY\s*=|Authorization:\s*Bearer|x-api-key\s*:)",
    re.IGNORECASE,
)


def _scan_paths() -> list[Path]:
    roots = [Path(".harness"), Path("logs")]
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        files.extend(p for p in root.rglob("*") if p.is_file())
    db = Path(".harness/harness.db")
    if db.exists():
        files.append(db)
    return files


def test_no_secrets_in_harness_artifacts() -> None:
    offenders: list[str] = []
    for path in _scan_paths():
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if FORBIDDEN.search(text):
            offenders.append(str(path))
    assert not offenders, f"secret-like content in: {offenders}"
