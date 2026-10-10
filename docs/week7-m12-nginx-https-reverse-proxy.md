# Week 7 M12: Nginx reverse proxy, HTTPS, public ingress

W7-M12 puts host-level Nginx in front of the Gradio UI and makes it the only
public entry point, at <https://agenticrag.hbapps.dedyn.io>. No application,
agent, or Compose behaviour changes; every container port stays on loopback.

## Architecture

```text
Internet
  -> deSEC DNS: agenticrag.hbapps.dedyn.io  A  169.58.163.248
  -> Nginx (host, systemd)   :80  -> 301 to HTTPS (and ACME challenges)
                             :443 -> TLS, Let's Encrypt certificate
  -> http://127.0.0.1:7860   Gradio container
  -> http://api:8000         FastAPI, over the Compose network only
```

```text
systemd
  docker.service
  production-agentic-rag.service   -> Docker Compose stack (M11)
  nginx.service                    -> public ingress (this milestone)
  certbot.timer                    -> certificate renewal
```

Nginx is not in Compose. It and the stack restart independently: Nginx
answers 502 while Gradio is down and recovers by itself when Gradio is back.

## DNS

deSEC (<https://desec.io/>) is authoritative for `hbapps.dedyn.io`. One `A`
record, `agenticrag` -> `169.58.163.248`, was created by hand in the deSEC
console. Nothing on the host talks to deSEC: there is no dynamic update, no
API token, and the certificate is issued over HTTP-01, not DNS-01. There is
no `AAAA` record, so the site is IPv4-only.

## Files

| Repository | Installed |
|---|---|
| `deploy/nginx/agenticrag.hbapps.dedyn.io.conf` (final, HTTPS) | `/etc/nginx/sites-available/agenticrag.hbapps.dedyn.io.conf`, symlinked from `sites-enabled/` |
| `deploy/nginx/agenticrag.hbapps.dedyn.io.http.conf` (bootstrap, HTTP only) | same path, until the `tls` stage replaces it |
| `deploy/nginx/install.sh` | not installed; run from the repository |

The installed file is a copy, not a symlink into the repository, so an edit
in the working tree cannot change the live server until it has passed
`nginx -t`. After editing the repository file, re-run the `tls` stage.

Certificates, keys, and the Let's Encrypt account live only under
`/etc/letsencrypt/`. None of it is in the repository.

## Installation

```bash
sudo deploy/nginx/install.sh http
curl -I http://agenticrag.hbapps.dedyn.io/      # 200 from Gradio, via Nginx

sudo CERTBOT_EMAIL=you@example.com deploy/nginx/install.sh tls
curl -I http://agenticrag.hbapps.dedyn.io/      # 301 to https
curl -I https://agenticrag.hbapps.dedyn.io/     # 200
```

`http` installs `nginx` from apt if missing, enables it, installs the
bootstrap config, and removes the packaged `sites-enabled/default` symlink
(it is a second `default_server` and serves a public welcome page).

`tls` installs `certbot` if missing, requests a certificate from the
production Let's Encrypt CA with the webroot method
(`/var/www/letsencrypt`), installs a renewal hook that reloads Nginx, and
switches to the final config. `CERTBOT_EMAIL` is optional; without it the
account has no expiry-notice address.

Both stages run `nginx -t` before reloading and restore the previous config
if it fails. Both can be re-run safely. Neither touches Docker, the
firewall, or DNS.

Certbot's webroot method is used instead of its Nginx plugin so that
Certbot never rewrites the server config; the repository file stays the
whole truth.

## Proxy behaviour

Only `location /` exists, and it proxies only to `http://127.0.0.1:7860`.
There is no location, upstream, or hostname for FastAPI, Airflow,
OpenSearch, Dashboards, PostgreSQL, or Redis. A request for `/docs`,
`/redoc`, `/api/...`, or `/airflow` reaches Gradio and gets Gradio's 404.
Gradio publishes its own `/openapi.json` and `/gradio_api/info`, which
describe the UI's event endpoints, not the FastAPI service.

Requests that do not carry the public hostname (the bare IP, scanners) hit
a catch-all server: port 80 closes the connection (`444`) and port 443
refuses the TLS handshake. HTTP for the real hostname is a `301` to HTTPS.

Headers: `X-Content-Type-Options`, `X-Frame-Options: SAMEORIGIN`,
`Referrer-Policy`. There is no Content-Security-Policy; Gradio relies on
inline scripts and styles. There is no `Strict-Transport-Security` yet: the
hostname may still change, and a browser would hold the policy long after.
`server_tokens` is off.

### Gradio behind the proxy

Gradio 6.15 needs no `root_path` when served at `/`. It builds the URLs it
hands to the browser from `X-Forwarded-Host` and `X-Forwarded-Proto`, so the
config sends both; with them `/config` reports
`root: https://agenticrag.hbapps.dedyn.io`, which is what prevents
mixed-content requests. Gradio trusts these headers from any client, which
is acceptable only because port 7860 is reachable from the host alone.

### Streaming

Gradio 6 delivers events to the browser over server-sent events
(`protocol: sse_v3`), not WebSocket. The `Upgrade` / `Connection` headers
are forwarded anyway so an upgrade request would still work.

`proxy_buffering off`, `proxy_cache off`, and `gzip off` in the proxied
location mean each status event is sent on as it arrives. HTTP/2 is enabled
on 443 so a browser's event streams are not limited to six connections.

### Timeouts

| Directive | Value | Why |
|---|---|---|
| `proxy_connect_timeout` | 30s | Gradio is local; a slow connect means it is down |
| `proxy_read_timeout` | 600s | Longest allowed silence from Gradio |
| `proxy_send_timeout` | 600s | Symmetric with the read timeout |

An agent run that falls back to live arXiv (search, PDF download, Docling,
rerank, generation, grounding) has a budget of about 540 seconds. 600
seconds keeps the proxy just above that, so the application's own timeout
fires first and the user sees its message, not a bare 504. The read timeout
counts silence, not total duration, and Gradio sends a heartbeat every 15
seconds, so a healthy stream is never near it. Nothing is infinite.

## Ports

| Port | Bound to | Public | Service |
|---|---|---|---|
| 22 | 0.0.0.0, [::] | yes | SSH |
| 80 | 0.0.0.0, [::] | yes | Nginx: redirect, ACME |
| 443 | 0.0.0.0, [::] | yes | Nginx: HTTPS -> Gradio |
| 7860 | 127.0.0.1 | no | Gradio |
| 8000 | 127.0.0.1 | no | FastAPI |
| 8080 | 127.0.0.1 | no | Airflow |
| 5601 | 127.0.0.1 | no | OpenSearch Dashboards |
| 9200, 9600 | 127.0.0.1 | no | OpenSearch |
| 5432 | 127.0.0.1 | no | PostgreSQL |
| 6379 | 127.0.0.1 | no | Redis |

The loopback bindings in `compose.yml` are what keep the internal services
private; they do not depend on a host firewall. To reach an internal tool,
tunnel over SSH: `ssh -L 8080:127.0.0.1:8080 <host>`.

## Certificate renewal

The Ubuntu `certbot` package installs `certbot.timer`, which runs
`certbot renew` twice a day and renews within 30 days of expiry. Renewal
uses the same webroot, served by the port-80 server block, so it needs no
downtime. `/etc/letsencrypt/renewal-hooks/deploy/reload-nginx.sh` reloads
Nginx after a successful renewal.

```bash
systemctl is-enabled certbot.timer
systemctl list-timers certbot.timer
sudo certbot certificates
sudo certbot renew --dry-run        # uses the staging CA; leaves the live certificate alone
```

## Operations

| Action | Command |
|---|---|
| Test config | `sudo nginx -t` |
| Reload (no dropped connections) | `sudo systemctl reload nginx` |
| Restart | `sudo systemctl restart nginx` |
| Status | `systemctl status nginx --no-pager` |
| Access / error log | `/var/log/nginx/access.log`, `/var/log/nginx/error.log` |
| Apply an edited repo config | `sudo deploy/nginx/install.sh tls` |
| Take the site offline | `sudo systemctl stop nginx` |

A restart or reload cuts any stream in flight; the user has to resubmit.

## Troubleshooting

- **DNS.** `dig +short agenticrag.hbapps.dedyn.io @ns1.desec.io` and
  `@1.1.1.1` must both return `169.58.163.248`. If they differ, wait for the
  record's TTL; if the first is wrong, fix the record in deSEC.
- **`nginx -t` fails.** The message names the file and line. A missing
  certificate file means the `tls` stage has not issued one yet; run the
  `http` stage first. `install.sh` has already restored the previous config.
- **Certbot fails to issue.** The challenge is fetched over port 80 from the
  Internet. Check that DNS is right, that port 80 is open in the provider
  firewall, and that
  `curl http://agenticrag.hbapps.dedyn.io/.well-known/acme-challenge/x`
  returns a 404 from Nginx rather than timing out. Details are in
  `/var/log/letsencrypt/letsencrypt.log`. Let's Encrypt allows five failed
  validations per hostname per hour.
- **Renewal fails.** `sudo certbot renew --dry-run` reproduces it;
  `journalctl -u certbot.service` has the timer's last run.
- **502 Bad Gateway.** Gradio is not listening. `docker compose ps -a`, then
  `curl -I http://127.0.0.1:7860/`. It starts only after `api` is healthy,
  so a 502 lasts a minute or two after a stack restart.
- **504 Gateway Timeout.** Gradio accepted the request and then sent nothing
  for 600 seconds. Look at `docker compose logs --tail=200 gradio api`; the
  agent should have timed out on its own before this.
- **A long request stops early.** Check `/var/log/nginx/error.log` for
  `upstream timed out` (proxy) versus an application error in the UI
  (agent). A stream that ends at exactly 60 seconds means the installed
  config is not the repository one.
- **Page loads without styling or events never arrive.** Check
  `curl -s https://agenticrag.hbapps.dedyn.io/config | grep -o '"root":"[^"]*"'`;
  anything other than the `https://` public URL means the forwarded headers
  are missing.

## Acceptance (2026-10-10)

Checked before installation, with the repository config running in a
temporary `nginx:1.24` container bound to loopback test ports and a
throwaway certificate (both removed afterwards):

- `nginx -t` passes for both configs on Nginx 1.24, the Ubuntu 24.04 version.
- HTTP for the hostname returns `301` to `https://agenticrag.hbapps.dedyn.io`
  with the path and query preserved; HTTP by IP is closed; TLS by IP is refused.
- The page loads over HTTP/2 with the security headers; `/config` reports the
  `https://` root.
- The cached question "What biases were found when auditing open-source
  large language models for clinical triage?" returned `Cache: HIT`, a
  `LOCAL` source, "Grounding validation passed", and the execution path.
  Seven events arrived one by one over 0.25 s as `text/event-stream`.

Host installation, the Let's Encrypt certificate, and the restart checks are
recorded here once the two `install.sh` stages have been run.

## Risks and follow-ups

- The provider firewall cannot be inspected from the host. Ports 80 and 443
  must be allowed there for the site and for certificate renewal.
- UFW state is recorded at installation. With every internal port on
  loopback it is a second layer, not the thing keeping them private.
- The UI has no authentication or rate limit. Anyone with the URL can submit
  questions, each of which may spend Groq tokens and trigger a live arXiv
  fallback.
- Gradio's own API (`/gradio_api/...`) is public along with the page, since
  the page is built on it.
- HSTS is deliberately off while the hostname may change. Add it once the
  hostname is final.
- The default Airflow login and the database password in `compose.yml`
  noted in M11 are unchanged; they remain reachable only from the host.
