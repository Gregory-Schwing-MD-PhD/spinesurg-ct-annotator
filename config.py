"""
Runtime configuration, loaded from environment variables.

On Hugging Face Spaces, the following are auto-injected when
`hf_oauth: true` is set in README.md frontmatter:
  • OAUTH_CLIENT_ID
  • OAUTH_CLIENT_SECRET
  • SPACE_HOST

You must set manually in the Space's Secrets UI:
  • HF_TOKEN              (write scope on the target dataset)
  • SESSION_SECRET        (any high-entropy string)
  • SOURCE_DATASET        (e.g. "greg-schwing/CTSpinoPelvic1K")
  • TARGET_DATASET        (e.g. "greg-schwing/CTSpinoPelvic1K-annotations")
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Hugging Face
    hf_token: str
    source_dataset: str
    target_dataset: str

    # OAuth (injected by HF Spaces when hf_oauth is enabled)
    oauth_client_id: str
    oauth_client_secret: str
    oauth_scopes: str = "openid profile"

    # Session
    session_secret: str = "dev-only-change-me"
    dev_mode: bool = False

    # MONAI Label subprocess
    monai_port: int = 8000
    monai_app_path: str = "/workspace/apps/radiology"
    # The radiology sample app's main.py requires at least one --conf models
    # value or it exits at boot. 'deepedit' is the lightest option; weights
    # only download when an annotator actually invokes inference, so pure
    # manual annotation incurs no GPU / download cost. Switch to a spinesurg-
    # ct-nnunet bundle here once live pre-labeling is desired.
    monai_models: str = "deepedit"
    monai_boot_timeout_s: int = 120
    ohif_path: str = "/ohif/"

    # Workspace
    workspace: str = "/workspace"

    # Public port (HF Spaces enforces 7860)
    public_port: int = 7860


settings = Settings()
