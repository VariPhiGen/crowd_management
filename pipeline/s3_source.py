"""
pipeline/s3_source.py
=====================
Provides S3VideoSource — a thin wrapper around boto3 that streams camera
videos one at a time from an S3 bucket, deletes the local copy after each
video is processed, and is fully backward-compatible with local paths.

Usage in cameras.json
---------------------
Set the "source" field to an S3 URI:
    "source": "s3://my-bucket/japan/cam_1/"

Credentials (in priority order)
--------------------------------
1. config/aws_credentials.json         ← explicit keys (gitignored)
2. Environment variables:              AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
3. ~/.aws/credentials                  ← AWS CLI profile
4. EC2 / ECS IAM instance role         ← recommended for cloud deployments

Temp directory
--------------
Downloaded videos land in  output/tmp_s3/<cam_id>/  by default and are
deleted immediately after each video finishes processing.
"""

import json
import logging
import os
from pathlib import Path
from typing import Iterator, Optional, Union
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".ts"}


def _is_s3_uri(source: str) -> bool:
    return str(source).startswith("s3://")


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    """Return (bucket, prefix) from 's3://bucket/prefix/path/'."""
    parsed = urlparse(uri)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")
    return bucket, prefix


def _build_boto3_client(config_dir: Optional[str] = None):
    """
    Build a boto3 S3 client.

    Priority:
    1. config/aws_credentials.json  — explicit keys (local dev, gitignored)
    2. Environment variables        — AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
    3. IAM instance role            — automatic on EC2 with attached instance profile
    """
    import boto3

    cred_path = Path(config_dir) / "aws_credentials.json" if config_dir else None

    if cred_path and cred_path.exists():
        try:
            with open(cred_path) as f:
                creds = json.load(f)
            client = boto3.client(
                "s3",
                aws_access_key_id     = creds.get("aws_access_key_id"),
                aws_secret_access_key = creds.get("aws_secret_access_key"),
                region_name           = creds.get("region_name", "ap-northeast-1"),
            )
            logger.info("S3 client: loaded credentials from %s", cred_path)
            return client
        except Exception as exc:
            logger.warning("Could not load credentials from %s: %s — falling back to instance role", cred_path, exc)

    # Fall back to environment variables or IAM instance role (EC2 deployment)
    logger.info("S3 client: using environment / IAM instance role credentials")
    return boto3.client("s3", region_name="ap-northeast-1")


class S3VideoSource:
    """
    Downloads camera videos from S3 one at a time, processes each, then
    deletes the local copy to keep disk usage minimal.

    Parameters
    ----------
    s3_uri : str
        S3 URI of the folder containing videos, e.g.
        ``s3://my-bucket/japan/cam_1/``
    camera_id : str
        Used to create an isolated temp directory per camera.
    tmp_root : str | Path
        Root directory for temporary downloads.  Defaults to
        ``output/tmp_s3``.
    config_dir : str | None
        Directory that may contain ``aws_credentials.json``.
    """

    def __init__(
        self,
        s3_uri:    str,
        camera_id: str,
        tmp_root:  Union[str, Path] = "output/tmp_s3",
        config_dir: Optional[str] = None,
    ) -> None:
        self.s3_uri    = s3_uri
        self.camera_id = camera_id
        self.tmp_dir   = Path(tmp_root) / camera_id
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

        self._bucket, self._prefix = _parse_s3_uri(s3_uri)
        self._client = _build_boto3_client(config_dir)

        self._keys: list[str] = self._list_video_keys()
        logger.info(
            "[%s] S3VideoSource: found %d video(s) at %s",
            camera_id, len(self._keys), s3_uri,
        )

    # ── Public helpers ────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._keys)

    @property
    def video_keys(self) -> list[str]:
        """Sorted list of S3 object keys for this camera."""
        return list(self._keys)

    def iter_videos(self) -> Iterator[Path]:
        """
        Yield one local Path at a time.  After the caller is done with the
        path, call ``delete_local(path)`` to free disk space immediately.
        This generator downloads each video just before yielding it.
        """
        for key in self._keys:
            local_path = self._download(key)
            yield local_path

    def iter_videos_prefetch(self, ahead: int = 1) -> Iterator[Path]:
        """
        Like ``iter_videos()`` but downloads the *next* ``ahead`` videos in a
        background thread while the current video is being processed.

        This eliminates the S3 download stall between consecutive videos — the
        next file is ready immediately when the caller finishes the current one.

        Parameters
        ----------
        ahead : int
            How many videos to prefetch ahead (default 1 — enough for most
            cases; set higher only if processing is faster than download).
        """
        from concurrent.futures import ThreadPoolExecutor, Future
        from collections import deque

        keys = list(self._keys)
        if not keys:
            return

        pending: deque[Future] = deque()

        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="s3pre") as pool:
            # Kick off the first `ahead + 1` downloads immediately
            prefill = min(ahead + 1, len(keys))
            for i in range(prefill):
                pending.append(pool.submit(self._download, keys[i]))

            next_submit = prefill
            for _ in keys:
                # Start the next download in the background before yielding
                if next_submit < len(keys):
                    pending.append(pool.submit(self._download, keys[next_submit]))
                    next_submit += 1

                try:
                    local_path = pending.popleft().result()
                    yield local_path
                except Exception as exc:
                    # Yield the exception so the caller can log and skip
                    yield exc  # type: ignore[misc]

    def delete_local(self, path: Path) -> None:
        """
        Delete the locally downloaded copy of a video.
        Safe to call with any Path — silently skips if already deleted.
        """
        try:
            path.unlink(missing_ok=True)
            logger.info("[%s] Deleted local copy: %s", self.camera_id, path.name)
        except Exception as exc:
            logger.warning("[%s] Could not delete %s: %s", self.camera_id, path, exc)

    def get_total_s3_frames(self) -> int:
        """
        Quick estimate of total frame count by summing object sizes.
        Used only for progress display — not downloaded yet.
        """
        return len(self._keys)  # actual count done post-download per-file

    # ── Private helpers ───────────────────────────────────────────────────────

    def _list_video_keys(self) -> list[str]:
        """List and sort all video object keys under the S3 prefix."""
        keys: list[str] = []
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=self._prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if Path(key).suffix.lower() in _VIDEO_EXTS:
                    keys.append(key)
        return sorted(keys)

    def _download(self, key: str, max_retries: int = 4) -> Path:
        """
        Download one S3 object to the temp dir and return its local Path.
        Retries up to *max_retries* times with exponential backoff so that
        transient network blips during a long multi-day run don't abort
        an entire camera thread.
        """
        import time as _time

        filename   = Path(key).name
        local_path = self.tmp_dir / filename

        if local_path.exists():
            logger.info("[%s] Using cached local copy: %s", self.camera_id, filename)
            return local_path

        last_exc: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(
                    "[%s] Downloading s3://%s/%s … (attempt %d/%d)",
                    self.camera_id, self._bucket, key, attempt, max_retries,
                )
                # Remove partial file from previous attempt if it exists
                if local_path.exists():
                    local_path.unlink()
                self._client.download_file(self._bucket, key, str(local_path))
                logger.info("[%s] Download complete: %s", self.camera_id, filename)
                return local_path
            except Exception as exc:
                last_exc = exc
                wait = 2 ** attempt   # 2, 4, 8, 16 seconds
                logger.warning(
                    "[%s] Download failed (attempt %d/%d): %s — retrying in %ds",
                    self.camera_id, attempt, max_retries, exc, wait,
                )
                _time.sleep(wait)

        # All retries exhausted — re-raise so the caller can skip this file
        raise RuntimeError(
            f"[{self.camera_id}] Failed to download {key} after {max_retries} attempts"
        ) from last_exc
