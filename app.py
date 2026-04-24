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
import hashlib
import hmac
import json
import logging
import subprocess
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from authlib.integrations.starlette_client import OAuth
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
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
# API tokens for Slicer                                                       #
# --------------------------------------------------------------------------- #
# Slicer's MONAI Label module isn't a browser — it can't follow the OAuth
# redirect chain. Instead we issue a per-user API token, derived as an HMAC of
# the username with the session secret. Deterministic (no DB needed), bound
# to one user, revokable by rotating SESSION_SECRET. Users paste it into
# Slicer's "Authorization" field (Basic Auth: username + token-as-password).


def user_api_token(username: str) -> str:
    """Stable per-user API token derived from the session secret."""
    mac = hmac.new(
        settings.session_secret.encode("utf-8"),
        username.encode("utf-8"),
        hashlib.sha256,
    )
    # URL-safe base64-ish; first 32 hex chars is plenty of entropy for this.
    return mac.hexdigest()[:32]


def username_from_request(request: Request) -> str | None:
    """Resolve the acting user from either a session cookie (browser) or
    a Basic/Bearer auth header (Slicer). Returns username or None."""
    # 1. Session cookie (browser users who went through OAuth).
    user = request.session.get("user")
    if user:
        return user

    # 2. Authorization header (Slicer users).
    auth = request.headers.get("authorization", "")
    if auth.startswith("Basic "):
        import base64
        try:
            raw = base64.b64decode(auth[6:]).decode("utf-8", errors="replace")
            user_part, _, token = raw.partition(":")
        except Exception:
            return None
        if user_part and token and hmac.compare_digest(
            token, user_api_token(user_part)
        ):
            return user_part
    elif auth.startswith("Bearer "):
        # Bearer <username>:<token> — simple format, one header value.
        token_part = auth[7:].strip()
        if ":" in token_part:
            user_part, token = token_part.split(":", 1)
            if user_part and hmac.compare_digest(
                token, user_api_token(user_part)
            ):
                return user_part
    return None


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
    """Return (username, session_id) or raise 401.

    Accepts either a session cookie (browser users) or Basic/Bearer
    Authorization (Slicer users). For API-token callers we synthesize a
    session_id from the request — it's only used for audit correlation
    and doesn't need to match a real session record.
    """
    user = request.session.get("user")
    sid = request.session.get("session_id")
    if user and sid:
        return user, sid

    api_user = username_from_request(request)
    if api_user:
        # Deterministic per-request "session" identifier: prefix plus a
        # truncated uuid so audit rows can correlate a Slicer save burst.
        return api_user, f"slicer-{uuid.uuid4().hex[:12]}"

    raise HTTPException(
        status_code=401,
        detail="Not authenticated",
        headers={"WWW-Authenticate": 'Basic realm="SpineSurg"'},
    )


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
    """Forward request to MONAI Label, streaming the response back.

    If MONAI Label is unreachable (crashed, still booting, or not yet
    configured), return a clean 503 instead of leaking an httpx traceback
    into the browser. The liveness probe in the lifespan hook sets the
    overall boot status; this is the per-request equivalent.
    """
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
    try:
        upstream = await HTTP_CLIENT.send(upstream_req, stream=True)
    except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError) as e:
        log.error("Upstream MONAI Label unreachable for %s %s: %s",
                  request.method, path, e)
        return JSONResponse(
            status_code=503,
            content={
                "error": "annotation_backend_unavailable",
                "detail": (
                    "MONAI Label is not responding. Check the Space's "
                    "container logs; the service may still be starting "
                    "or it may have crashed at boot."
                ),
            },
        )

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



# NOTE: The catch-all proxy route is defined at the BOTTOM of this file,
# after /, /open/{case_id}, and all other explicit handlers. FastAPI
# dispatches in registration order, so a catch-all declared here would
# shadow every explicit route defined later in the file (including the
# landing page at "/"). Keep it last.




# --------------------------------------------------------------------------- #
# Landing page — case picker                                                  #
# --------------------------------------------------------------------------- #

_LANDING_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>SpineSurg CT Annotator</title>
  <style>
    :root {
      --bg: #0f172a; --panel: #1e293b; --border: #334155;
      --ink: #e2e8f0; --muted: #94a3b8; --accent: #38bdf8;
      --accent-ink: #0f172a; --ok: #22c55e; --warn: #fbbf24;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; padding: 2.5rem 1.5rem; min-height: 100vh;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg); color: var(--ink); line-height: 1.55;
    }
    .wrap { max-width: 880px; margin: 0 auto; }
    header {
      display: flex; justify-content: space-between; align-items: baseline;
      border-bottom: 1px solid var(--border); padding-bottom: 1rem;
      margin-bottom: 2rem;
    }
    h1 { font-size: 1.5rem; margin: 0; font-weight: 600; }
    .user { font-size: 0.875rem; color: var(--muted); }
    .user a { color: var(--accent); text-decoration: none; margin-left: 0.75rem; }
    h2 { font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em;
         color: var(--muted); font-weight: 600; margin: 2.5rem 0 0.75rem; }
    .panel {
      background: var(--panel); border: 1px solid var(--border);
      border-radius: 8px; padding: 1.25rem 1.5rem; margin-bottom: 0.75rem;
    }
    .step-num {
      display: inline-block; width: 1.5rem; height: 1.5rem; line-height: 1.5rem;
      background: var(--accent); color: var(--accent-ink); border-radius: 50%;
      text-align: center; font-weight: 700; font-size: 0.8rem; margin-right: 0.5rem;
    }
    .url-box {
      display: flex; align-items: stretch; gap: 0; margin: 0.75rem 0;
      border: 1px solid var(--border); border-radius: 6px; overflow: hidden;
    }
    .url-box code {
      flex: 1; padding: 0.75rem 1rem; background: #000; color: var(--accent);
      font-size: 0.9rem; overflow-x: auto; white-space: nowrap;
      font-family: ui-monospace, SFMono-Regular, monospace;
    }
    .url-box button {
      padding: 0 1rem; background: var(--accent); color: var(--accent-ink);
      border: none; cursor: pointer; font-weight: 600; font-size: 0.85rem;
    }
    .url-box button:hover { filter: brightness(1.1); }
    .url-box button.copied { background: var(--ok); }
    a.dl {
      display: inline-block; padding: 0.6rem 1rem; background: var(--accent);
      color: var(--accent-ink); border-radius: 6px; text-decoration: none;
      font-weight: 600; font-size: 0.9rem; margin-top: 0.5rem;
    }
    a.dl:hover { filter: brightness(1.1); }
    .cases { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
             gap: 0.5rem; }
    .case {
      background: #0b1220; border: 1px solid var(--border); border-radius: 5px;
      padding: 0.5rem 0.75rem; font-family: ui-monospace, monospace;
      font-size: 0.82rem; color: var(--ink);
    }
    .empty {
      background: var(--panel); border: 1px dashed var(--border); border-radius: 6px;
      padding: 1.5rem; text-align: center; color: var(--muted); font-size: 0.9rem;
    }
    .empty code {
      background: #000; padding: 0.1rem 0.35rem; border-radius: 3px; color: var(--accent);
    }
    .status-line { font-size: 0.85rem; color: var(--muted); margin-top: 0.5rem; }
    .status-line.ok::before { content: "●  "; color: var(--ok); }
    .status-line.pending::before { content: "●  "; color: var(--warn); }
    p { margin: 0.5rem 0; }
    ul { margin: 0.5rem 0; padding-left: 1.25rem; }
    li { margin: 0.25rem 0; }
    .small { font-size: 0.82rem; color: var(--muted); }
    kbd {
      background: #000; border: 1px solid var(--border); border-radius: 3px;
      padding: 0.1rem 0.35rem; font-size: 0.8rem; font-family: ui-monospace, monospace;
    }
  </style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>SpineSurg CT Annotator</h1>
    <div class="user">
      signed in as <strong>__USERNAME__</strong>
      <a href="/logout">sign out</a>
    </div>
  </header>

  <h2>Setup (one time)</h2>

  <div class="panel">
    <p><span class="step-num">1</span><strong>Install 3D Slicer</strong> (free, cross-platform).</p>
    <a class="dl" href="https://download.slicer.org/" target="_blank" rel="noopener">Download 3D Slicer →</a>
    <p class="small">Any stable release 5.0+ works. Install, then launch.</p>
  </div>

  <div class="panel">
    <p><span class="step-num">2</span><strong>Install the MONAI Label extension.</strong></p>
    <p>In Slicer: <kbd>View</kbd> → <kbd>Extensions Manager</kbd>, search for <strong>MONAI Label</strong>, install, restart Slicer.</p>
  </div>

  <div class="panel">
    <p><span class="step-num">3</span><strong>Connect Slicer to this server.</strong></p>
    <p>In Slicer, switch to the <em>MONAI Label</em> module (module dropdown). Paste this URL into the <em>MONAI Label server</em> field and click <kbd>▼</kbd> to connect:</p>
    <div class="url-box">
      <code id="server-url">__SERVER_URL__</code>
      <button onclick="copyUrl('server-url', 'copy-url-btn')" id="copy-url-btn">Copy</button>
    </div>
    <p class="small">When Slicer prompts for authentication, use:</p>
    <p class="small"><strong>Username:</strong> <code style="background:#000;padding:0.1rem 0.4rem;border-radius:3px;color:var(--accent)">__USERNAME__</code></p>
    <div class="url-box">
      <code id="api-token">__API_TOKEN__</code>
      <button onclick="copyUrl('api-token', 'copy-token-btn')" id="copy-token-btn">Copy token</button>
    </div>
    <p class="small">This token is personal to you — it's derived from your HF username and rotates if the project admin rotates the server secret. <strong>Don't share it.</strong></p>
  </div>

  <h2>Your assigned cases</h2>

  <div class="panel">
    __CASES_HTML__
    <p class="status-line __STAGE_CLASS__">__STAGE_STATUS__</p>
  </div>

  <h2>Workflow</h2>

  <div class="panel">
    <ul>
      <li>In Slicer's MONAI Label panel, click <kbd>Next Sample</kbd> to load an unlabeled case.</li>
      <li>The seed segmentation (from TotalSegmentator + CTSpine1K/CTPelvic1K fused labels) is loaded automatically. Your job is to <strong>refine</strong>, especially around the lumbosacral junction on LSTV cases.</li>
      <li>Use the <em>Segment Editor</em> tools: Paint, Erase, Threshold, Islands, Smoothing.</li>
      <li>Click <kbd>Submit Label</kbd> to save. Your mask is versioned as <code>&lt;your-username&gt;_&lt;timestamp&gt;.nii.gz</code> and auto-pushed to the private annotations dataset.</li>
      <li>Saves never overwrite another annotator's work — multiple readers on the same case give us inter-rater variability by design.</li>
    </ul>
  </div>

</div>

<script>
function copyUrl(targetId, btnId) {
  const text = document.getElementById(targetId).textContent;
  navigator.clipboard.writeText(text);
  const btn = document.getElementById(btnId);
  const original = btn.textContent;
  btn.textContent = 'Copied ✓';
  btn.classList.add('copied');
  setTimeout(() => {
    btn.textContent = original;
    btn.classList.remove('copied');
  }, 1800);
}
</script>
</body>
</html>
"""


def _render_landing(
    username: str,
    cases: list[str],
    server_url: str,
    api_token: str,
    stage_done: bool,
) -> str:
    if cases:
        cards = "\n".join(f'<span class="case">{c}</span>' for c in cases)
        cases_html = f'<div class="cases">{cards}</div>'
        stage_status = (
            f"{len(cases)} case{'s' if len(cases) != 1 else ''} ready in Slicer."
            if stage_done
            else f"Preparing {len(cases)} case{'s' if len(cases) != 1 else ''}… "
                 "refresh this page in a moment."
        )
        stage_class = "ok" if stage_done else "pending"
    else:
        cases_html = (
            '<div class="empty">'
            "No cases assigned yet. Ask the project admin to add you to "
            "<code>assignments.json</code>."
            "</div>"
        )
        stage_status = ""
        stage_class = ""
    return (
        _LANDING_TEMPLATE
        .replace("__USERNAME__", username)
        .replace("__SERVER_URL__", server_url)
        .replace("__API_TOKEN__", api_token)
        .replace("__CASES_HTML__", cases_html)
        .replace("__STAGE_CLASS__", stage_class)
        .replace("__STAGE_STATUS__", stage_status)
    )


# Module-level tracker of which case-sets have been staged already this boot.
# Key: tuple of sorted case_ids. Value: asyncio.Task (pending) or True (done).
_STAGE_TASKS: dict[tuple, object] = {}


async def _stage_cases_bg(case_ids: list[str]) -> None:
    """Download every case's CT + seed label from the source HF dataset.

    Slicer's MONAI Label module queries `/datastore/` to list available
    studies; anything already materialized in /workspace/raw_data by
    ensure_case() shows up immediately. Staging all assigned cases at
    login time means annotators click 'Next Sample' in Slicer and the
    case is instantly available — no per-case pre-fetch latency.
    """
    for cid in case_ids:
        try:
            await asyncio.to_thread(sync.ensure_case, cid)
            log.info("staged %s", cid)
        except Exception:
            log.exception("stage failed for %s", cid)


def _trigger_staging(case_ids: list[str]) -> bool:
    """Kick off staging for a case set if not already staged. Returns True
    if staging has completed (files are ready on disk)."""
    if not case_ids:
        return True
    key = tuple(sorted(case_ids))
    prior = _STAGE_TASKS.get(key)
    if prior is True:
        return True
    if prior is None:
        task = asyncio.create_task(_stage_cases_bg(case_ids))
        _STAGE_TASKS[key] = task

        def _done(t):
            # Whatever happens, mark staging as complete so a failed
            # fetch doesn't block the user forever — they can still
            # annotate whatever cases did make it.
            _STAGE_TASKS[key] = True

        task.add_done_callback(_done)
        return False
    # Task in progress — check if it's actually finished.
    if isinstance(prior, asyncio.Task) and prior.done():
        _STAGE_TASKS[key] = True
        return True
    return False


@app.get("/", response_class=HTMLResponse)
async def landing(request: Request):
    if "user" not in request.session:
        return RedirectResponse("/login")
    username = request.session["user"]
    if ASSIGNMENTS_PATH.exists():
        data = json.loads(ASSIGNMENTS_PATH.read_text())
        cases = list(dict.fromkeys(data.get(username, []) + data.get("*", [])))
    else:
        cases = []
    cases = [c for c in cases if not c.startswith("_")]

    # Kick off background staging of every assigned case so Slicer sees
    # them as soon as the annotator clicks Next Sample.
    stage_done = _trigger_staging(cases)

    # Build the server URL that Slicer's MONAI Label extension needs. It's
    # just the Space's external URL — FastAPI proxies /datastore, /info,
    # /infer etc. through to the MONAI Label subprocess.
    server_url = str(request.base_url).rstrip("/")

    api_token = user_api_token(username)

    return HTMLResponse(
        _render_landing(username, cases, server_url, api_token, stage_done),
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


# NOTE: The old `/open/{case_id}` click-through route is no longer needed.
# Slicer enumerates the datastore itself, so there's no per-case "open"
# handshake. All assigned cases are pre-staged by the landing page above.


# --------------------------------------------------------------------------- #
# Generic reverse proxy — MUST be registered last                             #
#                                                                             #
# Any @app.get / @app.post / @app.api_route declared AFTER this will never    #
# be reached, because Starlette dispatches routes in registration order and   #
# "/{path:path}" matches every URL (including "/", with path=""). All         #
# explicit handlers in this file — /, /login, /logout, /auth/callback,       #
# /healthz, /api/*, /open/{case_id}, PUT /datastore/label — are defined       #
# above this point on purpose.                                                #
# --------------------------------------------------------------------------- #

@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
)
async def proxy_all(path: str, request: Request):
    # Belt-and-suspenders: refuse to proxy the root. The landing handler
    # owns "/" and should have caught it before we got here, but if
    # something upstream rewrites the request, fall through to the
    # landing page rather than leaking MONAI Label's root (Swagger) to
    # the user.
    if path == "":
        return RedirectResponse("/", status_code=307)

    # Anonymous allow-list: these paths should never be proxied.
    if path in PUBLIC_PATHS or path.startswith("auth/"):
        raise HTTPException(404)

    # Accept either a session cookie (browser) or an Authorization header
    # (Slicer). Slicer can't follow OAuth redirects, so unauthenticated
    # non-browser requests get a 401 with a WWW-Authenticate challenge
    # instead of a redirect.
    user = username_from_request(request)
    if user is None:
        ua = request.headers.get("user-agent", "")
        if "Mozilla" in ua or "Chrome" in ua or "Safari" in ua:
            return RedirectResponse(f"/login?next=/{path}")
        return JSONResponse(
            status_code=401,
            content={"detail": "Authentication required"},
            headers={"WWW-Authenticate": 'Basic realm="SpineSurg"'},
        )

    # Lazy-load CT volumes on first reference.
    case_id = request.query_params.get("image") or request.query_params.get("label")
    if case_id and request.method == "GET":
        try:
            await asyncio.to_thread(sync.ensure_case, case_id)
        except Exception as e:
            log.warning("ensure_case(%s) failed: %s", case_id, e)

    return await _proxy(request, path)
