#!/usr/bin/env bash
set -Eeuo pipefail

# Run on the AI EC2 instance with root privileges. Argument must be a full commit SHA.
sha=${1:?commit SHA required}
[[ "$sha" =~ ^[0-9a-f]{40}$ ]] || { echo 'Invalid commit SHA' >&2; exit 1; }
[[ $EUID -eq 0 ]] || { echo 'Run as root' >&2; exit 1; }
command -v flock >/dev/null
exec 9>/var/lock/safepath-ai-deploy.lock
flock -n 9 || { echo 'Another deployment is running' >&2; exit 1; }

repo=/home/ssm-user/safepath/safepath_AI
[[ -d "$repo/.git" ]] || { echo 'AI checkout missing' >&2; exit 1; }
# Build an isolated copy; never modify the existing checkout.
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
git -c safe.directory="$repo" -C "$repo" fetch origin "$sha"
git -c safe.directory="$repo" -C "$repo" archive "$sha" | tar -x -C "$tmp"
image="safepath-ai:$sha"
docker build --build-arg INSTALL_MODELS=1 -t "$image" "$tmp"

docker inspect safepath-ai >/dev/null
# Refuse to discard settings added since the initial setup.
docker inspect safepath-ai | python3 -c '
import json, sys
c = json.load(sys.stdin)[0]
allowed = {"AI_LOAD_MODELS", "AI_MODEL_CACHE_DIR", "HF_HOME", "AI_BACKEND_URL", "AI_BACKEND_KEY", "SSL_CERT_FILE", "NO_PROXY", "PATH", "LANG", "GPG_KEY", "PYTHON_VERSION", "PYTHON_SHA256", "PYTHONUNBUFFERED"}
unexpected = [e.split("=", 1)[0] for e in c["Config"]["Env"] if e.split("=", 1)[0] not in allowed]
if unexpected:
    print("Additional container settings require review:", ", ".join(unexpected), file=sys.stderr)
    sys.exit(1)
'
if docker inspect safepath-ai-previous >/dev/null 2>&1; then
  echo 'Previous rollback container exists; resolve it before deploying' >&2
  exit 1
fi
extra_args=()
if [[ -f /etc/safepath/ai-backend.env ]]; then
  [[ -f /etc/safepath/ai-trust-bundle.pem ]] || { echo 'AI backend TLS trust bundle missing' >&2; exit 1; }
  extra_args+=(--env-file /etc/safepath/ai-backend.env --mount type=bind,src=/etc/safepath/ai-trust-bundle.pem,dst=/run/safepath/ca-bundle.pem,readonly)
else
  if docker inspect safepath-ai --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -q '^AI_BACKEND_URL='; then
    echo 'Existing backend settings would be lost; stopping' >&2
    exit 1
  fi
fi
docker stop safepath-ai
if ! docker rename safepath-ai safepath-ai-previous; then
  docker start safepath-ai >/dev/null || true
  exit 1
fi
rollback() {
  echo 'Deployment failed; restoring previous container' >&2
  docker rm -f safepath-ai >/dev/null 2>&1 || true
  docker rename safepath-ai-previous safepath-ai || true
  docker start safepath-ai >/dev/null || true
}
trap 'rollback; rm -rf "$tmp"' ERR
docker run -d --name safepath-ai --restart unless-stopped \
  -p 127.0.0.1:8000:8000 \
  -e AI_LOAD_MODELS=1 -e AI_MODEL_CACHE_DIR=/cache/models \
  -e HF_HOME=/cache/huggingface -v safepath-model-cache:/cache "${extra_args[@]}" "$image" >/dev/null
for attempt in $(seq 1 120); do
  if curl -fsS --max-time 3 http://127.0.0.1:8000/health | python3 -c \
    'import json,sys; d=json.load(sys.stdin); sys.exit(0 if d.get("status")=="ok" and d.get("mode")=="model" and {"whisper","yamnet","speaker"}.issubset(d.get("models_loaded",[])) else 1)' 2>/dev/null; then
    docker rm safepath-ai-previous >/dev/null
    trap 'rm -rf "$tmp"' EXIT
    echo "Deployment succeeded: $sha"
    exit 0
  fi
  sleep 5
done
echo 'Model health check timed out' >&2
false
