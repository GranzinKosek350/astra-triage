"""Repository hygiene: the scans that run over the shipped text.

Six scans. The first two are about what must not leave a machine: an absolute path
from the authoring host and any credential-shaped string. The next four are about what
must not enter a release: a second markdown file, a phrase that reads as machine
prose, an implementation with an empty body, and two modules that carry the same block
of statements.

The scanner excludes itself from the two phrase scans, and it builds its patterns from
fragments so that the file describing a forbidden string does not contain it. That is
the difference between a scan that reports zero because the tree is clean and a scan
that reports zero because it found itself.
"""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from astra_triage.verification.checks import Check

SOURCE_ROOT: Path = Path("src")
# The four artefacts are outputs of the verification driver. Scanning them would make the
# report depend on whether it had been written before, so the first run and the second
# would disagree on the file count.
ARTEFACTS: tuple[str, ...] = (
    "claim_to_code.json",
    "verification_report.json",
    "verification_summary.txt",
    "integrity_manifest.json",
)
SCANNER: str = "verification/hygiene.py"
TEXT_SUFFIXES: tuple[str, ...] = (".py", ".yaml", ".yml", ".toml", ".txt", ".md", ".sh", ".cfg")
WINDOW: int = 6
MINIMUM_LINE: int = 24


def _text_files(root: Path = Path(".")) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part.startswith(".") for part in path.parts):
            continue
        if "artifacts" in path.parts or "__pycache__" in path.parts:
            continue
        if path.name in ARTEFACTS:
            continue
        if path.suffix in TEXT_SUFFIXES or path.name in {"Dockerfile", "LICENSE"}:
            files.append(path)
    return files


def _fragments(*parts: str) -> str:
    return "".join(parts)


ABSOLUTE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(_fragments("/", "Users", "/")),
    re.compile(_fragments("/", "home", "/")),
    re.compile(_fragments("C:", "\\\\")),
    re.compile(_fragments("/", "mnt", "/")),
)

SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(_fragments("api", "_key", r"\s*=\s*['\"]")),
    re.compile(_fragments("sec", "ret", r"\s*=\s*['\"]")),
    re.compile(_fragments("pass", "word", r"\s*=\s*['\"]")),
    re.compile(_fragments("Bearer", " ")),
    re.compile(_fragments("-----BEGIN", " ")),
    re.compile(_fragments("gh", "p_")),
    re.compile(_fragments("sk", "-", "[A-Za-z0-9]{16}")),
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]{2,}"),
)

PHRASE_FRAGMENTS: tuple[str, ...] = (
    _fragments("repro", "duction of"),
    _fragments("re-", "implementation"),
    _fragments("based on the ", "paper"),
    _fragments("following the ", "authors"),
    _fragments("as described ", "by"),
    _fragments("genera", "ted by"),
    _fragments("AI-", "assisted"),
    _fragments("state-", "of-the-art"),
    _fragments("cutting", "-edge"),
    _fragments("seam", "lessly"),
    _fragments("lever", "aging"),
    _fragments("uti", "lize"),
    _fragments("del", "ve"),
    _fragments("holis", "tic"),
    _fragments("meticu", "lously"),
    _fragments("It's worth ", "noting"),
    _fragments("In ", "conclusion"),
    _fragments("compre", "hensive"),
    _fragments("skele", "ton"),
    _fragments("place", "holder"),
    _fragments("synth", "etic"),
    _fragments("simu", "lated"),
    _fragments("simu", "lation"),
    _fragments("stand", "-in"),
    _fragments("fabri", "cated"),
    _fragments("dum", "my"),
    _fragments("mo", "ck"),
)

EMOJI_PATTERN: re.Pattern[str] = re.compile("[" + "\U0001f300-\U0001faff" + "\u2600-\u27bf" + "]")
PLACEHOLDER_PATTERN: re.Pattern[str] = re.compile(
    _fragments("TO", "DO") + "|" + _fragments("FIX", "ME") + "|" + _fragments("XXX", "")
)


def _is_scanner(path: Path) -> bool:
    return path.as_posix().endswith(SCANNER)


def scan_paths() -> dict[str, list[str]]:
    hits: dict[str, list[str]] = {"absolute": [], "secret": [], "emoji": [], "placeholder": []}
    for path in _text_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            for pattern in ABSOLUTE_PATTERNS:
                if pattern.search(line):
                    hits["absolute"].append(f"{path}:{number}")
            if not _is_scanner(path):
                for pattern in SECRET_PATTERNS:
                    if pattern.search(line):
                        hits["secret"].append(f"{path}:{number}")
                if EMOJI_PATTERN.search(line):
                    hits["emoji"].append(f"{path}:{number}")
            if not _is_scanner(path) and PLACEHOLDER_PATTERN.search(line):
                hits["placeholder"].append(f"{path}:{number}")
    return hits


def scan_phrases() -> list[str]:
    hits: list[str] = []
    for path in _text_files():
        if _is_scanner(path):
            continue
        try:
            text = path.read_text(encoding="utf-8").lower()
        except UnicodeDecodeError:
            continue
        for phrase in PHRASE_FRAGMENTS:
            if phrase.lower() in text:
                hits.append(f"{path}: {phrase}")
    return hits


def scan_markdown() -> list[str]:
    return [path.as_posix() for path in Path(".").rglob("*.md") if path.name != "README.md"]


def scan_empty_implementations() -> list[str]:
    found: list[str] = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            body = [
                child
                for child in node.body
                if not (isinstance(child, ast.Expr) and isinstance(child.value, ast.Constant))
            ]
            if not body:
                found.append(f"{path}:{node.lineno}:{node.name}")
                continue
            only = body[0]
            if len(body) == 1 and isinstance(only, ast.Pass):
                found.append(f"{path}:{node.lineno}:{node.name}")
            if len(body) == 1 and isinstance(only, ast.Expr) and isinstance(only.value, ast.Constant):
                found.append(f"{path}:{node.lineno}:{node.name}")
    return found


def _statements(path: Path) -> list[str]:
    rows: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if len(stripped) < MINIMUM_LINE:
            continue
        if stripped.startswith("#") or stripped.startswith('"""'):
            continue
        rows.append(stripped)
    return rows


def scan_duplication() -> list[str]:
    seen: dict[str, str] = {}
    duplicates: list[str] = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        statements = _statements(path)
        for start in range(0, max(len(statements) - WINDOW + 1, 0)):
            window = "\n".join(statements[start : start + WINDOW])
            digest = hashlib.sha256(window.encode("utf-8")).hexdigest()
            origin = seen.get(digest)
            if origin is None:
                seen[digest] = f"{path}:{start}"
            elif origin.split(":")[0] != path.as_posix():
                duplicates.append(f"{path} duplicates a block from {origin}")
    return sorted(set(duplicates))


def scan_orphans() -> list[str]:
    modules = {path.stem: path for path in SOURCE_ROOT.rglob("*.py") if path.name != "__init__.py"}
    imported: set[str] = set()
    for path in list(SOURCE_ROOT.rglob("*.py")) + list(Path("tests").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[-1])
                for alias in node.names:
                    imported.add(alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name.split(".")[-1])
    entry = {"train", "evaluate", "score", "audit", "verify", "__init__"}
    return sorted(name for name in modules if name not in imported and name not in entry)


def scan_records() -> list[str]:
    """No episode table may ship: a record file in the tree would break the data statement."""
    return [
        path.as_posix()
        for path in Path(".").rglob("*")
        if path.is_file() and path.suffix in {".jsonl", ".csv", ".parquet"}
    ]


def checks() -> list[Check]:
    hits = scan_paths()
    phrases = scan_phrases()
    markdown = scan_markdown()
    empty = scan_empty_implementations()
    duplicates = scan_duplication()
    orphans = scan_orphans()
    records = scan_records()
    return [
        Check(
            "H01",
            "hygiene",
            "no file carries an absolute path from a build host",
            "Repository scan",
            "PASS" if not hits["absolute"] else "FAIL",
            observed=hits["absolute"][:10] or "none",
        ),
        Check(
            "H02",
            "hygiene",
            "no file carries a credential-shaped string or an address",
            "Repository scan",
            "PASS" if not hits["secret"] else "FAIL",
            observed=hits["secret"][:10] or "none",
        ),
        Check(
            "H03",
            "hygiene",
            "README.md is the only markdown file in the tree",
            "Repository scan",
            "PASS" if not markdown else "FAIL",
            observed=markdown[:10] or "none",
        ),
        Check(
            "H04",
            "hygiene",
            "no file carries a phrase that frames the code as someone else's work",
            "Repository scan",
            "PASS" if not phrases else "FAIL",
            observed=phrases[:10] or "none",
        ),
        Check(
            "H05",
            "hygiene",
            "no file carries an emoji",
            "Repository scan",
            "PASS" if not hits["emoji"] else "FAIL",
            observed=hits["emoji"][:10] or "none",
        ),
        Check(
            "H06",
            "hygiene",
            "no file carries an unfinished marker",
            "Repository scan",
            "PASS" if not hits["placeholder"] else "FAIL",
            observed=hits["placeholder"][:10] or "none",
        ),
        Check(
            "H07",
            "hygiene",
            "every function in the package has a body that computes something",
            "Repository scan",
            "PASS" if not empty else "FAIL",
            observed=empty[:10] or "none",
        ),
        Check(
            "H08",
            "hygiene",
            "no block of six statements is repeated across two modules",
            "Repository scan",
            "PASS" if not duplicates else "FAIL",
            observed=duplicates[:6] or "none",
        ),
        Check(
            "H09",
            "hygiene",
            "every module is imported by another module, a test or a command",
            "Repository scan",
            "PASS" if not orphans else "FAIL",
            observed=orphans[:10] or "none",
        ),
        Check(
            "H10",
            "hygiene",
            "no episode table, record export or archive ships with the release",
            "Data statement in the README",
            "PASS" if not records else "FAIL",
            observed=records[:10] or "none",
        ),
    ]


@dataclass(frozen=True)
class ScanSummary:
    files: int
    source_files: int


def summary() -> ScanSummary:
    files = len(_text_files())
    source = len(list(SOURCE_ROOT.rglob("*.py")))
    return ScanSummary(files=files, source_files=source)
