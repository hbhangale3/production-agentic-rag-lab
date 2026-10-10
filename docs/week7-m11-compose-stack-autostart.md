# Week 7 M11: containerized Gradio, one Compose stack, autostart

W7-M11 moves Gradio into Docker, makes Docker Compose the single owner of the
whole application, and adds one systemd unit that brings the stack up at boot.
Nginx, a domain, and HTTPS are the next milestone; nothing here is public.

## Ownership

```text
systemd (production-agentic-rag.service)
  -> Docker
      -> one Compose project: production-agentic-rag-lab
           postgres, redis, opensearch, opensearch-dashboards, airflow, api, gradio
```

Compose owns dependency order, health checks, restart behaviour, networking,
and volumes. systemd only starts and stops the project as one unit.

## Services

All seven services are persistent. There are no separate one-shot containers:
Airflow's database initialisation and admin-user creation run inside the
`airflow` container's entrypoint each time it starts, before the webserver
and scheduler.

| Service | Container | Image | Host port | Waits for (healthy) | Health check |
|---|---|---|---|---|---|
| `postgres` | `rag-postgres` | `postgres:16-alpine` | 127.0.0.1:5432 | - | `pg_isready` |
| `redis` | `rag-redis` | `redis:7-alpine` | 127.0.0.1:6379 | - | `redis-cli ping` |
| `opensearch` | `rag-opensearch` | `opensearch:2.19.0` | 127.0.0.1:9200, 9600 | - | `/_cluster/health` |
| `opensearch-dashboards` | `rag-dashboards` | `opensearch-dashboards:2.19.0` | 127.0.0.1:5601 | opensearch | `/api/status` |
| `airflow` | `rag-airflow` | built from `airflow/` | 127.0.0.1:8080 | postgres, opensearch | `/health` reports metadata DB and scheduler healthy |
| `api` | `rag-api` | built from `Dockerfile` | 127.0.0.1:8000 | postgres, opensearch, redis | `/api/v1/health` |
| `gradio` | `rag-gradio` | same image as `api` | 127.0.0.1:7860 | api | HTTP GET `/` |

Every service has `restart: unless-stopped`. No health check needs the
Internet.

## Gradio container

Gradio is a main dependency of the project, so the `gradio` service runs the
API image with a different command (`python -m src.gradio_app`). There is one
dependency environment and one image to build.

- It calls the API at `http://api:8000`, the Compose service name. It does
  not use host loopback, `host.docker.internal`, or a container IP.
- It listens on `0.0.0.0:7860` inside the container
  (`GRADIO_SERVER_NAME`, `GRADIO_SERVER_PORT`); the host publishes that only
  on `127.0.0.1:7860`. Run outside Docker, `src.gradio_app` still defaults to
  `127.0.0.1`.
- It receives no `.env` file and no secrets: only the API URL and its own
  bind settings.
- It runs as an unprivileged user (`10001`), with all capabilities dropped
  and `no-new-privileges`. It mounts nothing, and logs go to stdout.
- Gradio's usage analytics are off, and no public share link is created.

## Ports

Every published port is bound to `127.0.0.1`. Before this milestone the API,
PostgreSQL, OpenSearch, Dashboards, and Airflow were published on `0.0.0.0`.
Containers still reach each other by service name on the `rag-network`
bridge; host tools keep using `localhost`. To reach a UI from a laptop, use
an SSH tunnel, for example `ssh -L 7860:127.0.0.1:7860 <host>`.

## Why services stayed down after the earlier reboot

`api`, `airflow`, and `opensearch-dashboards` had no restart policy, so
Docker did not start them at boot. The three services that did return
(`postgres`, `redis`, `opensearch`) already had `unless-stopped`. Airflow and
Dashboards were running normally until the host shut down; neither had an
application fault. Gradio was a host process started by hand.

## systemd unit

`deploy/systemd/production-agentic-rag.service` is the reference copy.

- `Requires=docker.service`, `After=docker.service network-online.target`.
- Runs as `harsh` in group `docker`, in the repository directory.
- Start: `docker compose up -d --wait --wait-timeout 600`. The unit becomes
  active only when every service is running and healthy.
- Stop: `docker compose stop --timeout 60`. Containers and volumes are kept.
- `Type=oneshot` with `RemainAfterExit=yes`, so the unit stays active while
  the stack is up.

Install:

```bash
sudo install -m 0644 deploy/systemd/production-agentic-rag.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now production-agentic-rag.service
```

At shutdown systemd stops the unit, which stops the containers. At boot the
enabled unit starts them in dependency order. If the host loses power
instead, Docker restarts the containers itself through their restart policy
and the unit's `up -d` then finds them already running.

## Operations

| Action | Command |
|---|---|
| Start | `sudo systemctl start production-agentic-rag.service` |
| Stop | `sudo systemctl stop production-agentic-rag.service` |
| Restart | `sudo systemctl restart production-agentic-rag.service` |
| Status | `sudo systemctl status production-agentic-rag.service` |
| Containers | `docker compose ps -a` |
| Logs | `docker compose logs --tail=200 <service>` |
| Rebuild after a code change | `docker compose build api gradio && docker compose up -d` |

Never run `docker compose down -v`: it deletes the PostgreSQL, OpenSearch,
Redis, and Airflow volumes.

## Troubleshooting

```bash
journalctl -u production-agentic-rag.service -b --no-pager
journalctl -u docker -b --no-pager
docker compose ps -a
docker compose logs --tail=200 <service>
docker inspect -f '{{.Name}} {{.HostConfig.RestartPolicy.Name}} {{.State.Health.Status}}' $(docker compose ps -aq)
```

- The unit failed to start: one service did not become healthy within ten
  minutes. `docker compose ps -a` shows which; the others are still running.
- Airflow is slow to become healthy: it initialises its database on every
  start and has a three-minute start period.
- Gradio shows a connection error: check that `api` is healthy; Gradio
  starts only after it.

## Acceptance (2026-10-10)

Stack built and started with the new definition; all checks below were run
against the running containers.

| Check | Result |
|---|---|
| `docker compose up -d --wait` | all 7 services healthy; 98 s first start, 54 s on a later start |
| Restart policy | `unless-stopped` on all 7 |
| Listening sockets | all 8 published ports on `127.0.0.1` only |
| PostgreSQL | accepting connections; 17 papers, as before |
| Redis | `PONG`; existing keys kept, none flushed |
| OpenSearch | `arxiv-papers` 17 docs, `arxiv-papers-chunks` 513 docs, as before |
| OpenSearch Dashboards | `/api/status` green, connected to OpenSearch |
| Airflow | metadata DB and scheduler healthy; `arxiv_paper_ingestion` (active) and `hello_world_week1` (paused) listed; no import errors; no DAG triggered |
| API | `/ping` and `/health` ok; cached agent question returns `hit` in about 5 ms |
| Gradio | page returns 200; submitting the cached question shows the answer, five `LOCAL` sources, `Cache: HIT`, and one "Validated cached answer found" step |
| Gradio container | reaches `http://api:8000`; runs as uid 10001; no secrets in its environment; no mounts; not privileged |
| Stop (`docker compose stop --timeout 60`) | all 7 stopped in 3 s; volumes intact |
| Start after stop | all 7 healthy again |

The OpenSearch cluster is `yellow` because `arxiv-papers` has a replica that
a single node cannot place; that predates this milestone.

The first agent request after the rebuild was a miss, not a hit: the earlier
container was still an older build, and the W7-M10 cache-identity change
means its entries are no longer matched. The answer was regenerated and
cached under the new identity.

**Not yet done:** installing the unit in `/etc/systemd/system`,
`systemctl stop/start/restart`, and the reboot test. They need `sudo`, which
was not available to the session that made these changes. The unit file
passes `systemd-analyze verify`, and its exact start and stop commands were
run by hand as the same user with the results above. Record the reboot
result here once it has been run.

### Resources with the stack idle

| | |
|---|---|
| Host memory | 11 GiB total, 4.1 GiB used, 7.5 GiB available |
| Swap | none |
| Disk | 193 GB, 66% used |
| `rag-opensearch` | 1.02 GiB |
| `rag-airflow` | 612 MiB |
| `rag-dashboards` | 297 MiB |
| `rag-api` | 128 MiB idle; about 2.6 GiB once BGE and Docling are loaded |
| `rag-gradio` | 112 MiB |
| `rag-postgres`, `rag-redis` | 31 MiB, 4 MiB |

## Risks and follow-ups

- No swap. With every service up and the API's models loaded, roughly 5 GiB
  is in use on a 12 GiB host. That is workable, but an Airflow ingestion run
  at the same time as a live Docling conversion has no cushion.
- The API and Gradio image is about 10.5 GB and takes about 11 minutes to
  rebuild after a source change, because source and the virtual environment
  are copied in one layer.
- The API container runs as root, as before.
- Airflow keeps its default `admin` / `admin` login and the database password
  is in `compose.yml`, as before. Both are reachable only from the host now,
  but should change before anything is proxied publicly.
- Airflow runs its webserver and scheduler in one container. A webserver
  crash is caught by the health check but does not restart the container;
  only a scheduler exit does.
