"""
Hugging Face dataset sync.

Two responsibilities:
  1. Lazy-download CT volumes from the SOURCE dataset on first access.
     (Avoids pulling hundreds of GB on every cold start.)
  2. On every save:
        a) Write a versioned mask to /workspace/labels/<case>/<user>_<iso>.nii.gz
        b) Audit-log the event.
        c) Fire-and-forget push of the mask + updated audit DB to the TARGET
           dataset via huggingface_hub.
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
    """Make a string safe for use in a filename."""
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

        # Serialize uploads to avoid rate-limit thrashing from concurrent
        # saves. Individual saves are already cheap; queuing pushes is fine.
        self._upload_lock = asyncio.Lock()

    # --------------------------------------------------------------------- #
    # Source dataset — lazy download                                        #
    # --------------------------------------------------------------------- #

    def ensure_case(self, case_id: str) -> Path:
        """Download ct.nii.gz for a case if not already cached. Idempotent."""
        safe_case = _sanitize(case_id)
        local_dir = self.raw / safe_case
        local_path = local_dir / "ct.nii.gz"

        if local_path.exists():
            return local_path

        local_dir.mkdir(parents=True, exist_ok=True)

        # Try a few plausible layouts; CTSpinoPelvic1K's hf_export is primary.
        candidate_filenames = [
            f"hf_export/{case_id}/ct.nii.gz",
            f"{case_id}/ct.nii.gz",
            f"images/{case_id}.nii.gz",
        ]
        last_err: Exception | None = None
        for filename in candidate_filenames:
            try:
                downloaded = hf_hub_download(
                    repo_id=self.source,
                    filename=filename,
                    repo_type="dataset",
                    token=self.token,
                )
                # Symlink (or copy) to our expected layout.
                if not local_path.exists():
                    try:
                        local_path.symlink_to(downloaded)
                    except OSError:
                        local_path.write_bytes(Path(downloaded).read_bytes())
                return local_path
            except EntryNotFoundError as e:
                last_err = e
                continue

        raise FileNotFoundError(
            f"Could not locate ct.nii.gz for case {case_id!r} in {self.source}. "
            f"Tried: {candidate_filenames}. Last error: {last_err}"
        )

    # --------------------------------------------------------------------- #
    # Target dataset — versioned saves                                      #
    # --------------------------------------------------------------------- #

    def _versioned_mask_path(self, case_id: str, username: str) -> Path:
        """labels/<case>/<user>_<ISO8601>.nii.gz — option 3."""
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
        """Persist mask locally, log the event, kick off the HF push."""
        out = self._versioned_mask_path(case_id, username)
        # Write via a thread so we don't block the event loop on large volumes.
        await asyncio.to_thread(out.write_bytes, mask_bytes)

        sha = self.audit.log_annotation(
            username=username,
            case_id=case_id,
            source_filename=f"{case_id}/ct.nii.gz",
            mask_filename=str(out.relative_to(self.workspace)),
            mask_bytes=mask_bytes,
            session_id=session_id,
        )

        # Fire-and-forget. We don't want OHIF to hang on HF's response time.
        asyncio.create_task(self._push_artifacts(out, sha, case_id, username))

        return {
            "path": str(out.relative_to(self.workspace)),
            "sha256": sha,
            "bytes": len(mask_bytes),
        }

    async def _push_artifacts(
        self, mask_path: Path, sha: str, case_id: str, username: str,
    ) -> None:
        """Push the mask + audit DB to the target dataset.

        Kept simple: two upload_file calls. For very high throughput we'd
        batch into upload_folder with a create_commit, but per-save commits
        give us clean per-annotation history.
        """
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
