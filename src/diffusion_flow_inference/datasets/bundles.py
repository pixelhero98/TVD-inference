from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence
from zipfile import ZipFile

from diffusion_flow_inference.common.paths import resolve_project_path

PROCESSED_DATASET_BUNDLE_MARKER = ".diffusion_flow_processed_dataset_bundle.json"
_ALLOWED_BUNDLE_ROOTS = {"data", "paper_datasets"}


def resolve_dataset_bundle(path: str | Path) -> Path:
    resolved = resolve_project_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Processed dataset bundle not found: {resolved}")
    return resolved


def _safe_bundle_members(zf: ZipFile) -> Iterable[Any]:
    for info in zf.infolist():
        if info.is_dir():
            continue
        member = PurePosixPath(info.filename)
        if info.filename.startswith("/") or ".." in member.parts:
            raise ValueError(f"Unsafe path in processed dataset bundle: {info.filename}")
        if not member.parts or member.parts[0] not in _ALLOWED_BUNDLE_ROOTS:
            raise ValueError(
                "Processed dataset bundles may only contain data/ and paper_datasets/ entries; "
                f"got {info.filename}"
            )
        yield info


def dataset_bundle_manifest(bundle_zip: str | Path) -> Dict[str, Any]:
    zip_path = resolve_dataset_bundle(bundle_zip)
    with ZipFile(zip_path) as zf:
        members = list(_safe_bundle_members(zf))
    files = [
        {
            "path": str(info.filename),
            "bytes": int(info.file_size),
            "compressed_bytes": int(info.compress_size),
        }
        for info in members
    ]
    names = [row["path"] for row in files]
    return {
        "bundle_zip": str(zip_path),
        "bundle_bytes": int(zip_path.stat().st_size),
        "file_count": int(len(files)),
        "uncompressed_bytes": int(sum(row["bytes"] for row in files)),
        "compressed_member_bytes": int(sum(row["compressed_bytes"] for row in files)),
        "contains_data_root": any(name.startswith("data/") for name in names),
        "contains_paper_dataset_root": any(name.startswith("paper_datasets/") for name in names),
        "files": files,
    }


def _bundle_signature(zip_path: Path) -> Dict[str, Any]:
    stat = zip_path.stat()
    return {
        "bundle_zip": str(zip_path),
        "bundle_bytes": int(stat.st_size),
        "bundle_mtime_ns": int(stat.st_mtime_ns),
    }


def _existing_extraction_is_current(extract_root: Path, signature: Mapping[str, Any], manifest: Mapping[str, Any]) -> bool:
    marker_path = extract_root / PROCESSED_DATASET_BUNDLE_MARKER
    if not marker_path.exists():
        return False
    if not (extract_root / "data").exists() or not (extract_root / "paper_datasets").exists():
        return False
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not all(marker.get(key) == value for key, value in signature.items()):
        return False
    return all((extract_root / str(row["path"])).exists() for row in manifest.get("files", []))


def extract_processed_dataset_bundle(
    bundle_zip: str | Path,
    extract_root: str | Path,
    *,
    force: bool = False,
) -> Dict[str, Any]:
    zip_path = resolve_dataset_bundle(bundle_zip)
    root = resolve_project_path(extract_root)
    signature = _bundle_signature(zip_path)
    manifest = dataset_bundle_manifest(zip_path)
    if not manifest["contains_data_root"] or not manifest["contains_paper_dataset_root"]:
        raise ValueError("Processed dataset bundles must contain both data/ and paper_datasets/ roots.")
    if not force and _existing_extraction_is_current(root, signature, manifest):
        return {
            "bundle_zip": zip_path,
            "extract_root": root,
            "data_root": root / "data",
            "paper_dataset_root": root / "paper_datasets",
            "manifest": manifest,
            "extracted": False,
        }

    root.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path) as zf:
        for info in _safe_bundle_members(zf):
            target = (root / PurePosixPath(info.filename).as_posix()).resolve()
            try:
                target.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"Unsafe extraction target in processed dataset bundle: {info.filename}") from exc
            zf.extract(info, path=root)

    marker = dict(signature)
    marker.update(
        {
            "file_count": int(manifest["file_count"]),
            "uncompressed_bytes": int(manifest["uncompressed_bytes"]),
        }
    )
    (root / PROCESSED_DATASET_BUNDLE_MARKER).write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    return {
        "bundle_zip": zip_path,
        "extract_root": root,
        "data_root": root / "data",
        "paper_dataset_root": root / "paper_datasets",
        "manifest": manifest,
        "extracted": True,
    }


def _resolve_existing_candidate(path: str | Path | None) -> Optional[Path]:
    if path is None:
        return None
    raw = str(path).strip()
    if not raw:
        return None
    return resolve_project_path(raw)


def ensure_processed_dataset_bundle(
    *,
    bundle_zip: str | Path | None,
    extract_root: str | Path,
    mode: str = "auto",
    dataset_root: str | Path | None = None,
    data_paths: Sequence[str | Path | None] = (),
) -> Optional[Dict[str, Any]]:
    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in {"auto", "extract", "none"}:
        raise ValueError(f"Unknown dataset bundle mode: {mode}")
    if normalized_mode == "none" or not str(bundle_zip or "").strip():
        return None

    resolved_dataset_root = _resolve_existing_candidate(dataset_root)
    resolved_data_paths = [path for path in (_resolve_existing_candidate(p) for p in data_paths) if path is not None]
    missing_inputs = False
    if resolved_dataset_root is not None and not resolved_dataset_root.exists():
        missing_inputs = True
    if any(not path.exists() for path in resolved_data_paths):
        missing_inputs = True
    if normalized_mode == "auto" and not missing_inputs:
        return None
    return extract_processed_dataset_bundle(bundle_zip, extract_root, force=False)
