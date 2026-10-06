# Render deployment repair

## Verification performed (2026-10-06)

- Built the Python 3.11 Linux image from this Dockerfile and requirements.
  As uid 10001, `/usr/local/bin/uvx` executed version 0.11.26; uv executed
  the same pinned version. `pip check` passed; `/app/.env` was absent.
- All 10 unittest cases passed in Linux. Provider responses and error paths in
  those tests are mocked; the existing backend import uses real PostgreSQL.
  After the access-logging correction, all six focused deployment tests passed
  on Windows and Linux, including Uvicorn's actual AccessFormatter and mapping
  arguments with numeric formatting. Real Linux requests to `/`, `/favicon.ico`
  and `/health?tavilyApiKey=<placeholder>` returned 200 with correctly formatted
  access logs, redacted credentials, and no logging errors.
- Live Windows and Linux API smoke checks used real Groq, Tavily and PostgreSQL.
  Linux returned actual hotel research, paused at human approval, and resumed
  the persisted thread successfully. Travel and approval both returned 200.
- Started the default image command with PORT=8123; GET /health passed over
  real HTTP. Travel and approval passed over that listener too, using the
  real `message`/`thread_id` and approval schemas. The live selection was
  hotel_agent + itinerary_agent. Live flights, weather, rejection/revision,
  and the hosted Render service were not tested.
- MCP discovery is inapplicable to the current REST revision. Historical MCP
  servers were inspected in Git but were not restored or integration-tested.
- Unrelated warnings: Windows Starlette TestClient/httpx deprecation, build-time
  pip root-user warning, and the existing global PostgreSQL connection's
  ResourceWarning on test process exit. No dependency conflict was found.

The logged `initialize_mcp()` path is from an older revision. Current commit
`ef55339` replaced MCP with direct REST providers. No AGENTS.md, Render YAML,
build stages, MCP configuration or npx launcher exists in this checkout.
Do not deploy the old revision or restore MCP just to reproduce its exception.

Historical configuration (commit `495008b`):

| Server | Transport | Executable / arguments | Environment / dependencies |
| --- | --- | --- | --- |
| Tavily | streamable_http | `https://mcp.tavily.com/mcp/?tavilyApiKey=<placeholder>` | TAVILY_API_KEY; langchain-mcp-adapters==0.3.0, mcp==1.28.1; no subprocess |
| AviationStack | stdio | `shutil.which('uvx') or 'uvx'`, `aviationstack-mcp` | inherited environment plus AVIATION_STACK_API_KEY (alias AVIATIONSTACK_API_KEY); uv and dynamically downloaded aviationstack-mcp |
| Weather | stdio | `sys.executable`, absolute repository path to custom_weather_mcp_server.py | inherited environment plus OPENWEATHER_API_KEY; mcp, requests, python-dotenv |

AviationStack requires uvx even when discovery is triggered by the hotel agent.
The exception establishes that the process could not resolve `uvx`; logs alone
cannot distinguish an absent installation from a PATH problem. The old Dockerfile
installed neither uv nor uvx. Current requirements pin uv==0.11.26, which supplies
both launchers for Docker and native pip installation. Docker checks them before
and after switching to the runtime user (uid 10001). No npx installation is needed.

Current REST requests have bounded timeouts/retries; MCP discovery is no longer
an application initialization step. PostgreSQL checkpoint setup and the graph's
human approval/resume behavior remain unchanged. deployment_checks.py reports
missing launchers as configuration errors and bounds version checks to 10 seconds.
The API reports configuration errors as 503 and hides internal exception text
for other failures. Application, httpx and urllib3 logging redact query credentials,
known environment secrets and exception tracebacks.

## Redeploy

1. Rotate the exposed Tavily key in Tavily, revoke the old key, and replace
   TAVILY_API_KEY in Render's Environment settings. Keep all secrets out of Git,
   build arguments and command lines. Restrict or remove retained exposed logs.
2. Commit and push these changes with the existing REST migration. In Render,
   select the correct repository and branch and confirm the deployed commit
   contains `ef55339` and this fix. A traceback mentioning mcp_client.py afterward
   means a different revision/service is still running.
3. Set Root Directory to the repository directory containing app.py and Dockerfile.
   If the Git repository itself starts there, leave Root Directory empty.
   For Docker runtime, use `./Dockerfile`, its default command, and `/health` as
   the health check. The command binds `0.0.0.0` and Render's PORT (8000 fallback).
4. For native Python runtime, select Python 3.11 and set Build Command to
   `pip install -r requirements.txt && uvx --version && python deployment_checks.py`.
   Set Start Command to `uvicorn app:app --host 0.0.0.0 --port $PORT`.
5. Configure DATABASE_URL (reachable PostgreSQL, SSL), GROQ_API_KEY,
   TAVILY_API_KEY, AVIATIONSTACK_API_KEY and OPENWEATHER_API_KEY. Preserve
   existing Groq token limits and DEFAULT_ORIGIN_IATA settings.
6. Select Manual Deploy > Clear build cache & deploy. Check build output for
   uv/uvx version checks and startup output for successful database initialization.
7. Check GET /health. POST /api/travel with
   `{"message":"Plan a 3-day Tokyo trip including hotels","thread_id":"render-smoke-unique"}`.
   Verify the draft and requires_approval, then POST /api/travel/approve with
   `{"thread_id":"render-smoke-unique","approved":true,"feedback":""}`.
   A health response alone does not verify the travel workflow or provider data.

## Repeatable local checks

```sh
docker build -t tripmate-deployment-check .
docker run --rm tripmate-deployment-check sh -c 'id && command -v uvx && uvx --version && python deployment_checks.py'
docker run --rm tripmate-deployment-check python -m unittest discover -s tests -p test_deployment.py -v
```

These focused tests mock the backend; they do not validate Groq, provider calls,
PostgreSQL or checkpoint resume. For a real run pass credentials via a protected
environment file at runtime (`--env-file`), publish port 8000, and exercise the
requests above. The Docker context excludes .env files and .git.

To test a running server rather than the in-process FastAPI client:
`python tests/smoke_live.py --base-url http://127.0.0.1:8000`.

Deployment references: [Render Docker](https://render.com/docs/docker),
[Render port binding](https://render.com/docs/web-services),
[uv installation](https://docs.astral.sh/uv/getting-started/installation/).
