"""Upload demo/index.html to a static Hugging Face Space. Token from HF_TOKEN only."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, create_repo

ROOT = Path(__file__).resolve().parent
REPO_ID = "nisaralll/udwm-distill-lab"


def main() -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN in the environment. Do not put it in a file.")
    create_repo(
        REPO_ID,
        repo_type="space",
        exist_ok=True,
        space_sdk="static",
        token=token,
        private=False,
    )
    staging = Path(tempfile.mkdtemp(prefix="udwm-space-"))
    (staging / "index.html").write_text((ROOT / "index.html").read_text(encoding="utf-8"), encoding="utf-8")
    (staging / "README.md").write_text((ROOT / "HF_README.md").read_text(encoding="utf-8"), encoding="utf-8")
    api = HfApi(token=token)
    api.upload_folder(
        folder_path=str(staging),
        repo_id=REPO_ID,
        repo_type="space",
        commit_message="UDWM distill lab: toy (w,g) field, not a SOTA agent",
    )
    print("https://huggingface.co/spaces/" + REPO_ID)


if __name__ == "__main__":
    main()
