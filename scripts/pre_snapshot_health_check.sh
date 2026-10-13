#!/usr/bin/env bash
set -u

PASS=0
FAIL=0

check() {
  local name="$1"
  shift
  if "$@" >/tmp/check.out 2>/tmp/check.err; then
    echo "PASS  $name"
    PASS=$((PASS+1))
  else
    echo "FAIL  $name"
    cat /tmp/check.err 2>/dev/null || true
    FAIL=$((FAIL+1))
  fi
}

echo "=================================================="
echo " PRE-SNAPSHOT VM HEALTH CHECK"
echo "=================================================="
echo

echo "== HOST =="
uptime
echo
free -h
echo
df -h /
echo

echo "== GIT =="
git status --short
git branch --show-current
git log -3 --oneline --decorate
echo

echo "== SYSTEMD =="
check "Docker active" systemctl is-active --quiet docker
check "Production RAG unit active" systemctl is-active --quiet production-agentic-rag.service
check "Nginx active" systemctl is-active --quiet nginx

echo
echo "Enabled status:"
systemctl is-enabled docker 2>/dev/null || true
systemctl is-enabled production-agentic-rag.service 2>/dev/null || true
systemctl is-enabled nginx 2>/dev/null || true

echo
echo "== DOCKER COMPOSE =="
docker compose ps
echo

# Verify expected containers are running
for svc in api gradio airflow postgres redis opensearch opensearch-dashboards; do
  cid="$(docker compose ps -q "$svc" 2>/dev/null)"
  if [ -n "$cid" ] && [ "$(docker inspect -f '{{.State.Running}}' "$cid" 2>/dev/null)" = "true" ]; then
    echo "PASS  container running: $svc"
    PASS=$((PASS+1))
  else
    echo "FAIL  container running: $svc"
    FAIL=$((FAIL+1))
  fi
done

echo
echo "== APPLICATION ENDPOINTS =="

check "FastAPI ping" \
  curl -fsS --max-time 10 http://127.0.0.1:8000/api/v1/ping

check "FastAPI health" \
  curl -fsS --max-time 15 http://127.0.0.1:8000/api/v1/health

check "Gradio local" \
  curl -fsS --max-time 15 http://127.0.0.1:7860/

check "Airflow local" \
  curl -fsS --max-time 15 http://127.0.0.1:8080/

check "OpenSearch" \
  curl -fsS --max-time 15 http://127.0.0.1:9200/

check "OpenSearch Dashboards" \
  curl -fsS --max-time 15 http://127.0.0.1:5601/

check "Redis ping" \
  docker compose exec -T redis redis-cli ping

check "Postgres readiness" \
  docker compose exec -T postgres pg_isready

echo
echo "== PUBLIC HTTPS =="

check "HTTPS public site" \
  curl -fsS --max-time 20 https://agenticrag.hbapps.dedyn.io/

check "HTTP redirects to HTTPS" \
  bash -c '
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 http://agenticrag.hbapps.dedyn.io/)
    [ "$code" = "301" ] || [ "$code" = "308" ]
  '

echo
echo "== TLS CERTIFICATE =="

echo | openssl s_client \
  -connect agenticrag.hbapps.dedyn.io:443 \
  -servername agenticrag.hbapps.dedyn.io \
  2>/dev/null | openssl x509 -noout -subject -issuer -dates

echo
echo "== OPEN PORTS =="
ss -ltnp | grep -E ':(22|80|443|5432|5601|6379|7860|8000|8080|9200|9600)\b' || true

echo
echo "== CONTAINER RESOURCE USAGE =="
docker stats --no-stream

echo
echo "== RECENT FAILED SYSTEMD UNITS =="
systemctl --failed --no-pager || true

echo
echo "=================================================="
echo "RESULT: $PASS passed / $FAIL failed"
echo "=================================================="

if [ "$FAIL" -eq 0 ]; then
  exit 0
else
  exit 1
fi
