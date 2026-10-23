#!/usr/bin/env python3
"""Fail closed when a code release includes unexpected or sensitive files.

The optional identifier list is private, remains outside the repository, and is
never echoed in the report. Passing this check supplements manual review; it
cannot establish that an arbitrary file is anonymous.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


ROOT_FILES = {
    "README.md", "pyproject.toml", "requirements-reproducible.txt", ".gitignore",
}
PATH_RULES = (
    re.compile(r"src/healthy_reconstruction/[a-z_][a-z_0-9]*\.py\Z"),
    re.compile(r"tests/test_[a-z_0-9]+\.py\Z"),
    re.compile(r"docs/[a-z_0-9-]+\.md\Z"),
    re.compile(r"config/[a-z_0-9-]+\.example\.json\Z"),
    re.compile(r"tools/audit_release\.py\Z"),
    re.compile(r"\.github/workflows/tests\.yml\Z"),
)
TEXT_RULES = {
    "local_absolute_path": re.compile(r"/(?:home|media|Users|mnt)/[A-Za-z0-9_.-]+"),
    "windows_local_path": re.compile(r"[A-Za-z]:\\(?:Users|Data|Documents)\\", re.I),
    "file_uri": re.compile(r"file:\x2f\x2f", re.I),
    "private_key": re.compile(r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----"),
    "github_token": re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})"),
    "api_secret": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{24,}"),
    "cloud_access_key": re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    "calendar_date": re.compile(r"\b(?:19|20)\d{2}[-_/](?:0[1-9]|1[0-2])[-_/](?:0[1-9]|[12]\d|3[01])\b"),
}
EMAIL = re.compile(r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def allowed_path(name: str) -> bool:
    return name in ROOT_FILES or any(rule.fullmatch(name) for rule in PATH_RULES)


def selected_files(root: Path, staged: bool) -> list[str]:
    # Audit the index before a commit, and committed/tracked files in automation.
    # The file contents audited below are exactly the index blobs when staged.
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=True, stdout=subprocess.PIPE,
    )
    return sorted(x.decode("utf-8") for x in result.stdout.split(b"\0") if x)


def audit(root: Path, staged: bool = False, private_identifiers: Path | None = None,
          reviewed_findings: Path | None = None) -> dict:
    findings = []
    identifiers = []
    if private_identifiers is not None:
        resolved = private_identifiers.resolve()
        if resolved.is_relative_to(root.resolve()):
            raise ValueError("The private identifier dictionary must remain outside the release.")
        identifiers = sorted({s.strip() for s in resolved.read_text().splitlines() if s.strip()}, key=len, reverse=True)
    forbidden = (
        re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(re.escape(s) for s in identifiers) + r")(?![A-Za-z0-9])", re.I)
        if identifiers else None
    )
    reviewed = set()
    if reviewed_findings is not None:
        if reviewed_findings.resolve().is_relative_to(root.resolve()):
            raise ValueError("Keep private review records outside the release.")
        review = json.loads(reviewed_findings.read_text())
        for item in review.get("reviewed_findings", []):
            if (item.get("rule") == "private_identifier"
                    and item.get("classification") == "semantic_false_positive"
                    and item.get("reason")):
                reviewed.add((item["file"], item["line"], item["line_sha256"]))
    reviewed_matches = 0
    file_hashes = {}
    files = selected_files(root, staged)
    if not files:
        findings.append({"file": ".", "rule": "empty_release"})
    for name in files:
        if not allowed_path(name):
            findings.append({"file": name, "rule": "path_not_allowlisted"})
            continue
        path = root / name
        mode = subprocess.run(["git", "-C", str(root), "ls-files", "--stage", "--", name], check=True, capture_output=True, text=True).stdout.split()[0]
        if mode not in {"100644", "100755"}:
            findings.append({"file": name, "rule": "non_regular_git_entry"})
            continue
        blob = (subprocess.run(["git", "-C", str(root), "show", ":" + name], check=True, stdout=subprocess.PIPE).stdout if staged else path.read_bytes())
        file_hashes[name] = hashlib.sha256(blob).hexdigest()
        try:
            content = blob.decode("utf-8")
        except UnicodeDecodeError:
            findings.append({"file": name, "rule": "binary_or_non_utf8"})
            continue
        if b"\0" in blob:
            findings.append({"file": name, "rule": "nul_byte"})
        if len(blob) > 250_000:
            findings.append({"file": name, "rule": "unexpectedly_large_source_file"})
        for line_number, line in enumerate(content.splitlines(), 1):
            rules = dict(TEXT_RULES)
            if forbidden is not None:
                rules["private_identifier"] = forbidden
            for rule, pattern in rules.items():
                if pattern.search(line):
                    line_hash = hashlib.sha256(line.encode("utf-8")).hexdigest()
                    if rule == "private_identifier" and (name, line_number, line_hash) in reviewed:
                        reviewed_matches += 1
                    else:
                        findings.append({"file": name, "line": line_number, "rule": rule})
            for match in EMAIL.finditer(line):
                if not match.group(0).lower().endswith(("@example.com", "@example.invalid")):
                    findings.append({"file": name, "line": line_number, "rule": "email_address"})
    return {
        "passed": not findings,
        "files_checked": len(files),
        "private_dictionary_used": bool(identifiers),
        "individually_reviewed_token_collisions": reviewed_matches,
        "findings": findings,
        "sha256": file_hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--staged", action="store_true", help="Audit exact staged blobs instead of working files.")
    parser.add_argument("--private-identifiers", type=Path)
    parser.add_argument("--reviewed-findings", type=Path,
                        help="Private semantic false-positive review; exact line hashes are required.")
    parser.add_argument("--report", type=Path, help="Optional report destination outside the public repository.")
    args = parser.parse_args()
    if args.report is not None and args.report.resolve().is_relative_to(args.root.resolve()):
        parser.error("Save audit reports outside the public repository.")
    report = audit(args.root, args.staged, args.private_identifiers, args.reviewed_findings)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    # Do not print the dictionary or matched confidential text.
    print(json.dumps({k: v for k, v in report.items() if k != "sha256"}, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
