"""Deploy the NeuroSymbolic agent dashboard to Hugging Face Spaces.

Required environment variable:
  HF_TOKEN        Hugging Face write token.

Optional environment variables:
  HF_SPACE_NAME   Space name when deploying under the token owner's account.
  HF_SPACE_REPO_ID Full repo id, for example username/NeuroSymbolic-Meta-Reasoner.
  HF_PRIVATE      Set to true to create/update a private Space.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi


DEFAULT_SPACE_NAME = "NeuroSymbolic-Meta-Reasoner"


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_token() -> str:
    token = os.getenv("HF_TOKEN", "").strip()
    if token:
        return token

    token = input("Enter Hugging Face write token: ").strip()
    if not token:
        print("HF_TOKEN is required for deployment.", file=sys.stderr)
        raise SystemExit(1)
    return token


def _resolve_repo_id(api: HfApi, token: str) -> str:
    explicit_repo_id = os.getenv("HF_SPACE_REPO_ID", "").strip()
    if explicit_repo_id:
        return explicit_repo_id

    space_name = os.getenv("HF_SPACE_NAME", DEFAULT_SPACE_NAME).strip() or DEFAULT_SPACE_NAME
    try:
        username = api.whoami(token=token)["name"]
    except Exception as exc:
        print(f"Could not read Hugging Face user profile: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    return f"{username}/{space_name}"


def main() -> int:
    project_root = Path(__file__).resolve().parent
    token = _get_token()
    api = HfApi()
    repo_id = _resolve_repo_id(api, token)
    private = _env_bool("HF_PRIVATE")

    print(f"Deploying to Hugging Face Space: https://huggingface.co/spaces/{repo_id}")
    print("Creating or updating public Space." if not private else "Creating or updating private Space.")

    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="gradio",
            private=private,
            exist_ok=True,
            token=token,
        )

        api.upload_folder(
            folder_path=str(project_root),
            repo_id=repo_id,
            repo_type="space",
            ignore_patterns=[
                ".env",
                ".git/**",
                ".github/**",
                ".pytest_cache/**",
                "__pycache__/**",
                "**/__pycache__/**",
                "**/*.pyc",
                "*.egg-info/**",
                "build/**",
                "cache/**",
                "data/**",
                "dist/**",
                "logs/**",
                "models/**",
                "venv/**",
                ".venv/**",
                "**/.DS_Store",
            ],
            token=token,
        )
    except Exception as exc:
        print(f"Deployment failed: {exc}", file=sys.stderr)
        return 1

    print(f"Deployment complete: https://huggingface.co/spaces/{repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
