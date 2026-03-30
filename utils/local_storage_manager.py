"""
Local Storage Manager for OmniDoc
Fallback/compatibility layer using local directory (mirrors BlobStorageManager API).

Hierarchical path structure (mirrors blob):
  <base_dir>/knowledge-base/<orgId>/<userId>/<sessionId>/<runId>/<stage>/<fileId>_<name>.<ext>
"""
import os
import re
import uuid
import json
import shutil
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

VALID_STAGES = ("input", "intermediate", "output")


def _sanitize(name: str, max_len: int = 120) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._\-]", "_", name)
    return safe[:max_len]


class LocalStorageManager:
    """Manages file storage on local disk with hierarchical org/user/session/run paths."""

    def __init__(self, base_dir: str = "tmp"):
        self.base_dir = base_dir
        self.base_directory = os.getenv("AZ_BLOB_BASE_DIRECTORY", "knowledge-base")
        os.makedirs(self.base_dir, exist_ok=True)
        logger.info(f"📁 Local storage initialized: {self.base_dir}/")

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------
    def _build_local_path(
        self,
        org_id: str,
        user_id: str,
        session_id: str,
        run_id: str,
        stage: str,
        file_id: str,
        original_filename: str,
    ) -> str:
        _, ext = os.path.splitext(original_filename)
        if not ext:
            ext = ".bin"
        safe_name = _sanitize(os.path.splitext(original_filename)[0])
        filename = f"{file_id}_{safe_name}{ext}"
        storage_dir = os.path.join(
            self.base_dir, self.base_directory,
            org_id or "unknown_org",
            user_id or "unknown_user",
            session_id or "no_session",
            run_id or "no_run",
            stage,
        )
        os.makedirs(storage_dir, exist_ok=True)
        return os.path.join(storage_dir, filename)

    def _run_dir(self, org_id: str, user_id: str, session_id: str, run_id: str) -> str:
        return os.path.join(
            self.base_dir, self.base_directory,
            org_id or "unknown_org",
            user_id or "unknown_user",
            session_id or "no_session",
            run_id or "no_run",
        )

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------
    def upload_file(
        self,
        file_data: bytes,
        original_filename: str,
        org_id: str = None,
        user_id: str = None,
        session_id: str = None,
        run_id: str = None,
        stage: str = "input",
        metadata: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        if stage not in VALID_STAGES:
            raise ValueError(f"Invalid stage '{stage}'. Must be one of {VALID_STAGES}")

        try:
            file_id = str(uuid.uuid4())
            local_path = self._build_local_path(
                org_id, user_id, session_id, run_id, stage, file_id, original_filename,
            )

            with open(local_path, "wb") as f:
                f.write(file_data)

            # Write metadata sidecar
            meta = metadata or {}
            meta.update({
                "original_filename": original_filename,
                "uploaded_at": datetime.utcnow().isoformat(),
                "org_id": org_id or "",
                "user_id": user_id or "",
                "session_id": session_id or "",
                "run_id": run_id or "",
                "stage": stage,
            })
            meta_path = local_path + ".meta.json"
            with open(meta_path, "w") as mf:
                json.dump(meta, mf, indent=2)

            logger.info(f"📤 Saved locally: {original_filename} → {local_path} ({len(file_data)} bytes) [{stage}]")

            return {
                "file_id": file_id,
                "blob_path": local_path,
                "blob_url": f"file://{os.path.abspath(local_path)}",
                "size_bytes": len(file_data),
                "extension": os.path.splitext(original_filename)[1] or ".bin",
                "uploaded_at": meta["uploaded_at"],
                "stage": stage,
            }

        except Exception as e:
            logger.error(f"❌ Local file save failed for {original_filename}: {e}")
            raise

    def download_file(self, blob_path: str) -> bytes:
        try:
            if not os.path.isabs(blob_path):
                blob_path = os.path.join(self.base_dir, blob_path)
            with open(blob_path, "rb") as f:
                file_data = f.read()
            logger.info(f"📥 Read locally: {blob_path} ({len(file_data)} bytes)")
            return file_data
        except FileNotFoundError:
            logger.error(f"❌ File not found: {blob_path}")
            raise
        except Exception as e:
            logger.error(f"❌ Error reading file: {e}")
            raise

    def download_to_local(self, blob_path: str, local_dir: str = None) -> str:
        """For local storage the file is already local, just return the path."""
        if not os.path.isabs(blob_path):
            blob_path = os.path.join(self.base_dir, blob_path)
        return blob_path

    def list_run_files(
        self,
        org_id: str,
        user_id: str,
        session_id: str,
        run_id: str,
        stage: str = None,
    ) -> List[Dict[str, Any]]:
        try:
            run_dir = self._run_dir(org_id, user_id, session_id, run_id)
            if stage:
                run_dir = os.path.join(run_dir, stage)
            if not os.path.exists(run_dir):
                return []

            file_list = []
            for root, _, files in os.walk(run_dir):
                for fname in files:
                    if fname.endswith(".meta.json"):
                        continue
                    fpath = os.path.join(root, fname)
                    meta = {}
                    meta_path = fpath + ".meta.json"
                    if os.path.exists(meta_path):
                        with open(meta_path) as mf:
                            meta = json.load(mf)
                    file_list.append({
                        "blob_path": fpath,
                        "size_bytes": os.path.getsize(fpath),
                        "created_at": datetime.fromtimestamp(os.path.getctime(fpath)).isoformat(),
                        "metadata": meta,
                    })
            return file_list
        except Exception as e:
            logger.error(f"❌ Error listing files: {e}")
            return []

    def generate_sas_url(self, blob_path: str, expiry_hours: int = 24) -> str:
        if not os.path.isabs(blob_path):
            blob_path = os.path.join(self.base_dir, blob_path)
        return f"file://{os.path.abspath(blob_path)}"

    def blob_exists(self, blob_path: str) -> bool:
        if not os.path.isabs(blob_path):
            blob_path = os.path.join(self.base_dir, blob_path)
        return os.path.exists(blob_path)
