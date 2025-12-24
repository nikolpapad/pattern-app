from dataclasses import dataclass
from pathlib import Path
from typing import List

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}

@dataclass(frozen=True)
class Pattern:
    id: str
    name: str
    filename: str


def load_patterns(pattern_dir: str) -> List[Pattern]:
    base = Path(pattern_dir)
    if not base.exists():
        raise RuntimeError(f"Pattern directory not found: {pattern_dir}")

    patterns: List[Pattern] = []

    for file in sorted(base.iterdir()):
        if file.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        pattern_id = file.stem.lower().replace(" ", "_")
        name = file.stem.replace("_", " ").title()

        patterns.append(
            Pattern(
                id=pattern_id,
                name=name,
                filename=file.name,
            )
        )

    return patterns


def list_patterns(pattern_dir: str) -> List[dict]:
    return [
        {"id": p.id, "name": p.name}
        for p in load_patterns(pattern_dir)
    ]


def resolve_pattern_path(pattern_dir: str, pattern_id: str) -> Path:
    patterns = load_patterns(pattern_dir)

    for p in patterns:
        if p.id == pattern_id:
            return Path(pattern_dir) / p.filename

    raise KeyError(f"Unknown pattern_id: {pattern_id}")
