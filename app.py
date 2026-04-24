"""
SpineSurg CT Annotator — FastAPI front door.

Architecture:
  • FastAPI is the sole listener on the Space's public port (7860).
  • MONAI Label runs as a child subprocess bound to 127.0.0.1:MONAI_PORT.
  • Every inbound request passes through FastAPI, which:
      (1) enforces Hugging Face OAuth (required before any OHIF asset loads),
      (2) lazy-downloads CT volumes from the source HF dataset on first access,
      (3) intercepts label-save requests to persist versioned masks AND push
          them to the target HF dataset,
      (4) proxies everything else to MONAI Label unchanged.
"""
from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from authlib.integrations.starlette_client import OAuth
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from starlette.middleware.sessions import SessionMiddleware

from audit import AuditLogger
from config import settings
from sync_manager import SyncManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("annotator")

# --------------------------------------------------------------------------- #
# Globals initialised at module load; lifespan hooks manage runtime state.    #
# --------------------------------------------------------------------------- #

oauth = OAuth()
oauth.register(
    name="huggingface",
    client_id=settings.oauth_client_id,
    client_secret=settings.oauth_client_secret,
    server_metadata_url="https://huggingface.co/.well-known/openid-configuration",
    client_kwargs={"scope": settings.oauth_scopes},
)

audit = AuditLogger(Path(settings.workspace) / "audit.sqlite")
sync = SyncManager(
    source_dataset=settings.source_dataset,
    target_dataset=settings.target_dataset,
    workspace=settings.workspace,
    hf_token=settings.hf_token,
    audit_logger=audit,
)

MONAI_PROC: subprocess.Popen | None = None
HTTP_CLIENT: httpx.AsyncClient | None = None

ASSIGNMENTS_PATH = Path(__file__).parent / "assignments.json"

# Endpoints that bypass the auth wall.
PUBLIC_PATHS: set[str] = {"login", "logout", "auth/callback", "healthz", "favicon.ico"}


# --------------------------------------------------------------------------- #
# Lifespan                                                                    #
# --------------------------------------------------------------------------- #


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Boot MONAI Label, open a shared HTTP client, tear both down on exit."""
    global MONAI_PROC, HTTP_CLIENT

    studies_dir = Path(settings.workspace) / "raw_data"
    studies_dir.mkdir(parents=True, exist_ok=True)

    log.info("Starting MONAI Label subprocess on port %d", settings.monai_port)
    log.info("  app=%s  studies=%s  models=%s",
             settings.monai_app_path, studies_dir, settings.monai_models or "(none)")
    if not Path(settings.monai_app_path).exists():
        log.error("MONAI Label app path does not exist: %s", settings.monai_app_path)
        log.error("Listing /opt to help diagnose:")
        try:
            for p in sorted(Path("/opt").rglob("sample-apps/*"))[:50]:
                log.error("  %s", p)
        except Exception as e:
            log.error("  (could not list /opt: %s)", e)

    # Build argv. We run in annotation-only mode by default: no model is
    # configured, so MONAI Label serves OHIF + the datastore without trying
    # to load any inference weights. Static seed labels from the source HF
    # dataset are what annotators refine. Live pre-labeling will be added
    # later by setting monai_models to a bundle name (e.g. a spinesurg-ct-
    # nnunet checkpoint) and bumping Space hardware to a GPU tier.
    monai_argv = [
        "monailabel", "start_server",
        "--app", settings.monai_app_path,
        "--studies", str(studies_dir),
        "--host", "127.0.0.1",
        "--port", str(settings.monai_port),
    ]
    if settings.monai_models:
        monai_argv += ["--conf", "models", settings.monai_models]

    # Inherit the parent's stdout/stderr so MONAI Label's own logs (and any
    # crash traceback) are visible in the HF container-logs view. Without
    # this the subprocess can fail silently behind a PIPE nobody reads.
    MONAI_PROC = subprocess.Popen(monai_argv)

    HTTP_CLIENT = httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{settings.monai_port}",
        timeout=httpx.Timeout(300.0, connect=10.0),
    )

    # Wait for MONAI Label's /info endpoint to respond before accepting traffic.
    for attempt in range(settings.monai_boot_timeout_s):
        try:
            r = await HTTP_CLIENT.get("/info/")
            if r.status_code == 200:
                log.info("MONAI Label is ready (attempt %d)", attempt)
                break
        except httpx.RequestError:
            pass
        await asyncio.sleep(1)
    else:
        log.error("MONAI Label failed to start within %ds", settings.monai_boot_timeout_s)

    try:
        yield
    finally:
        log.info("Shutting down")
        if HTTP_CLIENT:
            await HTTP_CLIENT.aclose()
        if MONAI_PROC:
            MONAI_PROC.terminate()
            try:
                MONAI_PROC.wait(timeout=10)
            except subprocess.TimeoutExpired:
                MONAI_PROC.kill()


app = FastAPI(title="SpineSurg CT Annotator", lifespan=lifespan)
# Session cookie config for OAuth on HF Spaces:
#   • same_site="none" because the OAuth callback from huggingface.co back to
#     <space>.hf.space is a cross-site redirect; SameSite=Lax drops the cookie
#     on the return leg, which manifests as MismatchingStateError.
#   • https_only=True is required once SameSite=None (browsers reject
#     insecure cross-site cookies).
# In local dev (settings.dev_mode=True) we relax both so http://localhost works.
app.add_middleware(
    SessionMiddleware,
    secret_key=settings.session_secret,
    max_age=60 * 60 * 8,  # 8-hour clinical session
    same_site="lax" if settings.dev_mode else "none",
    https_only=not settings.dev_mode,
)


# --------------------------------------------------------------------------- #
# Auth dependencies                                                           #
# --------------------------------------------------------------------------- #


def require_user(request: Request) -> tuple[str, str]:
    """Return (username, session_id) or raise 401."""
    user = request.session.get("user")
    sid = request.session.get("session_id")
    if not user or not sid:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user, sid


def user_has_case(username: str, case_id: str) -> bool:
    """Assignment enforcement. Under option 3 (versioned saves), *multiple*
    users MAY be assigned the same case — concurrent annotation is a feature,
    not a bug. Missing file = permissive mode (useful for local dev)."""
    if not ASSIGNMENTS_PATH.exists():
        return True
    try:
        data = json.loads(ASSIGNMENTS_PATH.read_text())
    except json.JSONDecodeError:
        log.warning("assignments.json is malformed; denying access")
        return False
    return case_id in data.get(username, []) or case_id in data.get("*", [])


# --------------------------------------------------------------------------- #
# Auth routes                                                                 #
# --------------------------------------------------------------------------- #


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.get("/login")
async def login(request: Request):
    redirect_uri = str(request.url_for("auth_callback"))
    return await oauth.huggingface.authorize_redirect(request, redirect_uri)


@app.get("/auth/callback", name="auth_callback")
async def auth_callback(request: Request):
    try:
        token = await oauth.huggingface.authorize_access_token(request)
    except Exception as e:
        log.exception("OAuth exchange failed")
        raise HTTPException(400, f"OAuth failure: {e}")

    userinfo = token.get("userinfo")
    if not userinfo:
        resp = await oauth.huggingface.get("oauth/userinfo", token=token)
        userinfo = resp.json()

    username = userinfo.get("preferred_username") or userinfo.get("name")
    if not username:
        raise HTTPException(400, "OAuth response missing username")

    session_id = str(uuid.uuid4())
    request.session["user"] = username
    request.session["session_id"] = session_id
    audit.log_session(username, "login", session_id)
    log.info("User %s logged in (session %s)", username, session_id[:8])
    return RedirectResponse("/")


@app.get("/logout")
async def logout(request: Request):
    user = request.session.get("user")
    sid = request.session.get("session_id")
    if user and sid:
        audit.log_session(user, "logout", sid)
    request.session.clear()
    return RedirectResponse("/login")


# --------------------------------------------------------------------------- #
# App API                                                                     #
# --------------------------------------------------------------------------- #


@app.get("/api/whoami")
async def whoami(user_session=Depends(require_user)):
    username, sid = user_session
    return {"username": username, "session_id": sid}


@app.get("/api/my-cases")
async def my_cases(user_session=Depends(require_user)):
    username, _ = user_session
    if not ASSIGNMENTS_PATH.exists():
        return {"cases": [], "permissive": True}
    data = json.loads(ASSIGNMENTS_PATH.read_text())
    assigned = list(dict.fromkeys(data.get(username, []) + data.get("*", [])))
    return {"cases": assigned, "permissive": False}


# --------------------------------------------------------------------------- #
# Save interception                                                           #
#                                                                             #
# MONAI Label's datastore accepts label saves at `PUT /datastore/label`.      #
# We intercept BEFORE the catch-all proxy to:                                 #
#   1. Authorize the user against assignments.                                #
#   2. Persist the mask to `/workspace/labels/<case>/<user>_<iso>.nii.gz`.    #
#   3. Audit-log with sha256 + session_id.                                    #
#   4. Kick off a background push to the target HF dataset.                   #
#   5. Forward the same bytes to MONAI Label so its internal datastore        #
#      state remains coherent for subsequent reads.                           #
# --------------------------------------------------------------------------- #


@app.put("/datastore/label")
async def save_label(request: Request, user_session=Depends(require_user)):
    username, session_id = user_session

    # MONAI Label passes the image/case identifier in the query string.
    case_id = request.query_params.get("label") or request.query_params.get("image")
    if not case_id:
        raise HTTPException(400, "Missing label/image identifier in query")

    if not user_has_case(username, case_id):
        audit.log_session(
            username, "unauthorized_save_attempt", session_id,
            case_id=case_id,
        )
        raise HTTPException(403, f"User {username!r} not assigned to case {case_id!r}")

    body = await request.body()
    if not body:
        raise HTTPException(400, "Empty request body")

    # 1) Persist locally + audit log + async push to target dataset.
    result = await sync.save_mask(
        case_id=case_id,
        username=username,
        mask_bytes=body,
        session_id=session_id,
    )

    # 2) Mirror to MONAI Label so its internal state is consistent.
    try:
        upstream = await HTTP_CLIENT.put(
            "/datastore/label",
            params=dict(request.query_params),
            content=body,
            headers={
                "content-type": request.headers.get(
                    "content-type", "application/octet-stream",
                ),
            },
        )
        upstream_status = upstream.status_code
    except httpx.RequestError as e:
        log.warning("Failed to mirror to MONAI Label: %s", e)
        upstream_status = 0

    return JSONResponse(
        {
            "ok": True,
            "case_id": case_id,
            "username": username,
            "saved_to": result["path"],
            "sha256": result["sha256"],
            "monai_status": upstream_status,
        }
    )


# --------------------------------------------------------------------------- #
# Generic reverse proxy                                                       #
# --------------------------------------------------------------------------- #


async def _proxy(request: Request, path: str) -> StreamingResponse:
    """Forward request to MONAI Label, streaming the response back."""
    upstream_url = httpx.URL(
        path=f"/{path}",
        query=request.url.query.encode("utf-8"),
    )

    # Strip hop-by-hop headers.
    forward_headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in {"host", "connection", "content-length"}
    }

    body = await request.body() if request.method not in {"GET", "HEAD"} else None

    upstream_req = HTTP_CLIENT.build_request(
        method=request.method,
        url=upstream_url,
        headers=forward_headers,
        content=body,
    )
    upstream = await HTTP_CLIENT.send(upstream_req, stream=True)

    response_headers = {
        k: v for k, v in upstream.headers.items()
        if k.lower() not in {
            "content-encoding", "transfer-encoding",
            "content-length", "connection",
        }
    }

    return StreamingResponse(
        upstream.aiter_raw(),
        status_code=upstream.status_code,
        headers=response_headers,
        background=upstream.aclose,
    )


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
)
async def proxy_all(path: str, request: Request):
    # Anonymous allow-list.
    if path in PUBLIC_PATHS or path.startswith("auth/"):
        raise HTTPException(404)

    if "user" not in request.session:
        # Preserve the user's target so we can redirect back after login.
        return RedirectResponse(f"/login?next=/{path}")

    # Lazy-load CT volumes on first reference.
    case_id = request.query_params.get("image") or request.query_params.get("label")
    if case_id and request.method == "GET":
        try:
            await asyncio.to_thread(sync.ensure_case, case_id)
        except Exception as e:
            log.warning("ensure_case(%s) failed: %s", case_id, e)

    return await _proxy(request, path)


# Root → OHIF index (served by MONAI Label at /ohif/).
@app.get("/")
async def root(request: Request):
    if "user" not in request.session:
        return RedirectResponse("/login")
    return RedirectResponse(settings.ohif_path)
