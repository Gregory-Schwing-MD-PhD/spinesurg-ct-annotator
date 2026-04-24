---
title: SpineSurg CT Annotator
emoji: 🦴
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
hf_oauth: true
hf_oauth_scopes:
  - email 
hf_oauth_expiration_minutes: 480
hf_oauth_redirect_path: /auth/callback
---

# SpineSurg CT Annotator

Web-based annotation platform for the CTSpinoPelvic1K dataset. Clinical
collaborators authenticate with their Hugging Face account, receive
case assignments, and draw segmentations in OHIF. Every save is versioned
per-annotator, audit-logged with SHA-256 + session ID, and pushed back to
a parallel HF dataset for provenance.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ HF Space container (port 7860 public)                        │
│                                                              │
│  ┌──────────────────┐     reverse proxy     ┌────────────┐  │
│  │ FastAPI (7860)   │ ────────────────────► │ MONAI Label│  │
│  │  • HF OAuth      │                        │  + OHIF    │  │
│  │  • assignments   │ ◄──── intercept ────── │ (127:8000) │  │
│  │  • save hook     │      PUT /datastore/   └────────────┘  │
│  └────────┬─────────┘         label                          │
│           │                                                  │
│           │  (background)                                    │
│           ▼                                                  │
│  ┌──────────────────────┐        ┌──────────────────────┐   │
│  │ audit.sqlite (WAL)   │        │ huggingface_hub push │   │
│  └──────────────────────┘        └──────────┬───────────┘   │
└───────────────────────────────────────────────┼─────────────┘
                                                │
                     ┌──────────────────────────┴──────────┐
                     ▼                                     ▼
          SOURCE dataset (read-only)           TARGET dataset (R/W)
          CTSpinoPelvic1K                      CTSpinoPelvic1K-annotations
          └─ hf_export/CASE/ct.nii.gz          └─ labels/CASE/user_ISO.nii.gz
                                                └─ audit.sqlite
```

## Concurrent annotation

Under the versioned-save scheme, two annotators can open the same case;
each save lands at a distinct path:

```
labels/<case_id>/<username>_<ISO8601>.nii.gz
```

This gives inter-rater variability for free — valuable for LSTV where
Castellvi classification is known to disagree across readers.

## Local development

```bash
cp .env.example .env           # fill in HF_TOKEN and OAUTH_* creds
docker build -t spinesurg-annotator .
docker run --rm -it \
  -p 7860:7860 \
  --env-file .env \
  spinesurg-annotator
```

## Deployment

Push to `main` on GitHub; the `sync-to-hf-space` workflow mirrors to the
Space's git repo and HF rebuilds the Docker image automatically.

## Case assignment

Edit `assignments.json` and commit. The `"*"` key is a broadcast list.

```json
{
  "greg-schwing": ["COLONOG-0001", "COLONOG-0002"],
  "collaborator-username": ["COLONOG-0001"],
  "*": []
}
```

Two users both listed for `COLONOG-0001` → concurrent annotation; both
saves are preserved independently.

## Future: ensemble pre-labeling

MONAI Label is started with `--conf models segmentation_spine`, which
is a no-op placeholder for the first iteration. When the nnU-Net
checkpoints from `spinesurg-ct-nnunet` are ready, add a new app under
`/opt/monailabel/sample-apps/radiology/lib/infers/` and update the
`--conf models` flag in `config.py`. No changes to the proxy layer
should be needed.
