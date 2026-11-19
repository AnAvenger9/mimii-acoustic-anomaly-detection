"""Strict path labels and deterministic, duplicate-aware recording splits."""

from collections import defaultdict
import hashlib
from pathlib import Path
import re


def parse_recording(path, data_root):
    path, root = Path(path), Path(data_root)
    relative = path.relative_to(root)
    if len(relative.parts) != 3:
        raise ValueError(f"Expected id_XX/condition/filename.wav: {relative}")
    machine_id, condition, filename = relative.parts
    if not re.fullmatch(r"id_\d{2}", machine_id):
        raise ValueError(f"Invalid machine ID: {machine_id}")
    if condition not in {"normal", "abnormal"} or Path(filename).suffix.lower() != ".wav":
        raise ValueError(f"Invalid recording path: {relative}")
    return {
        "relative_path": relative.as_posix(),
        "machine_type": root.name,
        "machine_id": machine_id,
        "condition": condition,
        "label": int(condition == "abnormal"),
        "filename": filename,
    }


def discover_recordings(data_root):
    root = Path(data_root)
    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".wav")
    if not paths:
        raise ValueError(f"No WAV recordings found under {root}")
    return [parse_recording(p, root) for p in paths]


def assign_splits(records, seed=42, validation_fraction=0.1, test_normal_fraction=0.2):
    """Assign full recordings, grouping identical selected-channel waveforms.

    Fraction targets are by unique waveform within each ID; duplicate groups stay
    together. Abnormal clips always go to test. Conflicting duplicate labels fail.
    """
    if not (0 < validation_fraction < 1 and 0 < test_normal_fraction < 1
            and validation_fraction + test_normal_fraction < 1):
        raise ValueError("Invalid split fractions")
    groups = defaultdict(list)
    for row in records:
        groups[row["waveform_sha256"]].append(row)
    by_id = defaultdict(list)
    for digest, rows in groups.items():
        if len({(r["machine_id"], r["label"]) for r in rows}) != 1:
            raise ValueError("Identical waveforms have conflicting labels or machine IDs")
        if rows[0]["label"] == 1:
            for row in rows:
                row["split"] = "test"
        else:
            by_id[rows[0]["machine_id"]].append(digest)
    for machine_id, digests in sorted(by_id.items()):
        ordered = sorted(digests, key=lambda d: hashlib.sha256(
            f"{seed}:{machine_id}:{d}".encode()).hexdigest())
        count = len(ordered)
        if count < 3:
            raise ValueError(f"{machine_id}: need at least three distinct normal recordings")
        n_test = max(1, int(count * test_normal_fraction))
        n_val = max(1, int(count * validation_fraction))
        if n_test + n_val >= count:
            raise ValueError(f"{machine_id}: no training recordings remain")
        for index, digest in enumerate(ordered):
            split = "test" if index < n_test else "validation" if index < n_test + n_val else "train"
            for row in groups[digest]:
                row["split"] = split
    validate_splits(records)
    return records


def validate_splits(records):
    paths, content_splits = set(), defaultdict(set)
    source_splits = defaultdict(set)
    for row in records:
        if row["relative_path"] in paths:
            raise ValueError("A recording appears more than once")
        paths.add(row["relative_path"])
        if row["split"] not in {"train", "validation", "test"}:
            raise ValueError("Unknown split")
        if row["split"] != "test" and row["label"] != 0:
            raise ValueError("Anomalous audio entered train/validation")
        content_splits[row["waveform_sha256"]].add(row["split"])
        if "source_recording_key" in row:
            source_splits[row["source_recording_key"]].add(row["split"])
    if any(len(splits) != 1 for splits in content_splits.values()):
        raise ValueError("Identical audio leaked across splits")
    if any(len(splits) != 1 for splits in source_splits.values()):
        raise ValueError("Noise versions of the same source recording leaked across splits")


def assign_reference_splits(records, reference_records):
    """Keep matching original recording paths in the same split across SNRs.

    Assumes the release preserves source-recording identity in ID/condition/name.
    Full path inventories and labels must match; waveform equality is not expected.
    """
    validate_splits(reference_records)
    reference = {row["relative_path"]: row for row in reference_records}
    if len({r["relative_path"] for r in records}) != len(records):
        raise ValueError("Duplicate paths in target SNR data")
    if set(reference) != {r["relative_path"] for r in records}:
        raise ValueError("SNR inventories differ; investigate before transferring splits")
    for row in records:
        original = reference[row["relative_path"]]
        if any(row[key] != original[key] for key in ("machine_type", "machine_id", "condition", "label")):
            raise ValueError("Recording labels do not align across SNRs")
        row["split"] = original["split"]
    validate_splits(records)
    return records
