"""
Hugging Face dataset sync.

Source dataset layout (CTSpinoPelvic1K, flat):
    ct/{case_id}_ct.nii.gz
    labels/{case_id}_label.nii.gz

where case_id follows export_hf.py's filename convention:
    {token:04d}_{position}                       # fused
    {token:04d}_{position}_spine                  # separate, spine view
    {token:04d}_{position}_pelvic                 # separate, pelvic view
    e.g. "0189_unknown", "0401_unknown_pelvic"

Target dataset layout (CTSpinoPelvic1K-annotations):
    labels/{case_id}/{username}_{ISO8601}.nii.gz  # versioned refinements
    audit.sqlite                                   # provenance DB

The source label is offered to OHIF as a starting segmentation so clinicians
refine rather than redraw. This matches the paper's thesis: TotalSegmentator's
automated labels have a 19.6 junction-DSC gap on LSTV cases, so the value of
the annotation pass is corrective.
"""
from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

log = logging.getLogger("sync_manager")

_SAFE = re.compile(r"[^A-Za-z0-9_\-]")


def _sanitize(s: str) -> str:
    return _SAFE.sub("_", s)


class SyncManager:
    def __init__(
        self,
        source_dataset: str,
        target_dataset: str,
        workspace: str,
        hf_token: str,
        audit_logger: Any,
    ):
        self.source = source_dataset
        self.target = target_dataset
        self.workspace = Path(workspace)
        self.raw = self.workspace / "raw_data"
        self.labels = self.workspace / "labels"
        self.audit_db = self.workspace / "audit.sqlite"

        self.raw.mkdir(parents=True, exist_ok=True)
        self.labels.mkdir(parents=True, exist_ok=True)

        self.token = hf_token
        self.api = HfApi(token=hf_token)
        self.audit = audit_logger

        # Serialize uploads to avoid rate-limit thrashing from concurrent saves.
        self._upload_lock = asyncio.Lock()

    # --------------------------------------------------------------------- #
    # Source dataset — lazy per-case download                               #
    # --------------------------------------------------------------------- #

    def ensure_case(self, case_id: str) -> dict:
        """
        Ensure CT + source label for a case are present locally. Idempotent.

        Returns paths so the caller (FastAPI) can pass them to MONAI Label's
        datastore for registration. The source label is the *starting* mask —
        OHIF loads it and the annotator refines on top.
        """
        safe = _sanitize(case_id)
        case_dir = self.raw / safe
        case_dir.mkdir(parents=True, exist_ok=True)

        local_ct = case_dir / "ct.nii.gz"
        local_seed_label = case_dir / "seed_label.nii.gz"

        ct_remote = f"ct/{case_id}_ct.nii.gz"
        label_remote = f"labels/{case_id}_label.nii.gz"

        if not local_ct.exists():
            try:
                downloaded = hf_hub_download(
                    repo_id=self.source,
                    filename=ct_remote,
                    repo_type="dataset",
                    token=self.token,
                )
                self._link_or_copy(Path(downloaded), local_ct)
            except EntryNotFoundError as e:
                raise FileNotFoundError(
                    f"CT not found for case {case_id!r} at {ct_remote} "
                    f"in {self.source}: {e}"
                ) from e

        # Seed label is optional — future datasets may not ship one.
        if not local_seed_label.exists():
            try:
                downloaded = hf_hub_download(
                    repo_id=self.source,
                    filename=label_remote,
                    repo_type="dataset",
                    token=self.token,
                )
                self._link_or_copy(Path(downloaded), local_seed_label)
            except EntryNotFoundError:
                log.info("No seed label for %s; annotator starts from blank", case_id)

        return {
            "ct": str(local_ct),
            "seed_label": str(local_seed_label) if local_seed_label.exists() else None,
            "case_dir": str(case_dir),
        }

    @staticmethod
    def _link_or_copy(src: Path, dst: Path) -> None:
        try:
            dst.symlink_to(src)
        except OSError:
            dst.write_bytes(src.read_bytes())

    # --------------------------------------------------------------------- #
    # Target dataset — versioned saves (option 3)                           #
    # --------------------------------------------------------------------- #

    def _versioned_mask_path(self, case_id: str, username: str) -> Path:
        """labels/<case>/<user>_<ISO8601>.nii.gz — concurrent-safe."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        case_dir = self.labels / _sanitize(case_id)
        case_dir.mkdir(parents=True, exist_ok=True)
        return case_dir / f"{_sanitize(username)}_{ts}.nii.gz"

    async def save_mask(
        self,
        case_id: str,
        username: str,
        mask_bytes: bytes,
        session_id: str,
    ) -> dict:
        """Persist refined mask locally, log the event, push to target dataset."""
        out = self._versioned_mask_path(case_id, username)
        await asyncio.to_thread(out.write_bytes, mask_bytes)

        sha = self.audit.log_annotation(
            username=username,
            case_id=case_id,
            source_filename=f"ct/{case_id}_ct.nii.gz",
            mask_filename=str(out.relative_to(self.workspace)),
            mask_bytes=mask_bytes,
            session_id=session_id,
        )

        # Fire-and-forget — OHIF shouldn't wait on HF's response time.
        asyncio.create_task(self._push_artifacts(out, sha, case_id, username))

        return {
            "path": str(out.relative_to(self.workspace)),
            "sha256": sha,
            "bytes": len(mask_bytes),
        }

    async def _push_artifacts(
        self, mask_path: Path, sha: str, case_id: str, username: str,
    ) -> None:
        """Upload the refined mask and the updated audit DB to target."""
        async with self._upload_lock:
            try:
                await asyncio.to_thread(
                    self.api.upload_file,
                    path_or_fileobj=str(mask_path),
                    path_in_repo=str(mask_path.relative_to(self.workspace)),
                    repo_id=self.target,
                    repo_type="dataset",
                    commit_message=(
                        f"annotation: {case_id} by {username} "
                        f"(sha256={sha[:12]})"
                    ),
                )
                if self.audit_db.exists():
                    await asyncio.to_thread(
                        self.api.upload_file,
                        path_or_fileobj=str(self.audit_db),
                        path_in_repo="audit.sqlite",
                        repo_id=self.target,
                        repo_type="dataset",
                        commit_message=f"audit: after {mask_path.name}",
                    )
                log.info("Pushed %s to %s", mask_path.name, self.target)
            except Exception:
                log.exception("Failed to push %s to %s", mask_path, self.target)
