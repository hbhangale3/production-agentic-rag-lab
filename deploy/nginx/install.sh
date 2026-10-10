#!/usr/bin/env bash
# Installs host-level Nginx as the public ingress for the Gradio UI, in two stages:
#
#   sudo deploy/nginx/install.sh http   # Nginx + HTTP-only proxy, to prove DNS -> Nginx -> Gradio
#   sudo deploy/nginx/install.sh tls    # Let's Encrypt certificate, then the final HTTPS config
#
# Set CERTBOT_EMAIL to register the Let's Encrypt account with an expiry-notice address;
# without it the account is registered with no email.
#
# The script never touches Docker, the firewall, or DNS. A config that fails `nginx -t`
# is rolled back and Nginx is left running on the previous one.
set -euo pipefail

DOMAIN=agenticrag.hbapps.dedyn.io
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
AVAILABLE=/etc/nginx/sites-available/$DOMAIN.conf
ENABLED=/etc/nginx/sites-enabled/$DOMAIN.conf
WEBROOT=/var/www/letsencrypt
RENEW_HOOK=/etc/letsencrypt/renewal-hooks/deploy/reload-nginx.sh

[[ $EUID -eq 0 ]] || { echo "run with sudo" >&2; exit 1; }

ensure_packages() {
    local missing=()
    for package in "$@"; do
        dpkg -s "$package" >/dev/null 2>&1 || missing+=("$package")
    done
    if ((${#missing[@]})); then
        apt-get update
        DEBIAN_FRONTEND=noninteractive apt-get install -y "${missing[@]}"
    fi
}

# Copy a repo config into place, test the whole Nginx configuration, and reload.
activate() {
    local source=$1 backup=""
    if [[ -f $AVAILABLE ]]; then
        backup=$(mktemp)
        cp -p "$AVAILABLE" "$backup"
    fi
    install -m 0644 "$source" "$AVAILABLE"
    ln -sfn "$AVAILABLE" "$ENABLED"
    # The packaged default site is a second default_server and serves a public welcome page.
    rm -f /etc/nginx/sites-enabled/default
    if ! nginx -t; then
        echo "nginx -t failed; rolling back" >&2
        if [[ -n $backup ]]; then
            cp -p "$backup" "$AVAILABLE"
        else
            rm -f "$ENABLED" "$AVAILABLE"
        fi
        exit 1
    fi
    systemctl enable --now nginx
    systemctl reload nginx
}

case "${1:-}" in
http)
    ensure_packages nginx
    install -d -m 0755 "$WEBROOT"
    activate "$HERE/$DOMAIN.http.conf"
    echo "HTTP proxy active. Check: curl -I http://$DOMAIN/"
    ;;
tls)
    ensure_packages nginx certbot
    install -d -m 0755 "$WEBROOT"
    if [[ ! -f /etc/letsencrypt/live/$DOMAIN/fullchain.pem ]]; then
        [[ -f $ENABLED ]] || { echo "run the http stage first" >&2; exit 1; }
        if [[ -n ${CERTBOT_EMAIL:-} ]]; then
            account=(--email "$CERTBOT_EMAIL")
        else
            account=(--register-unsafely-without-email)
        fi
        certbot certonly --webroot -w "$WEBROOT" -d "$DOMAIN" \
            --non-interactive --agree-tos "${account[@]}"
    fi
    # Nginx only picks up a renewed certificate on reload.
    install -d -m 0755 "$(dirname "$RENEW_HOOK")"
    printf '#!/bin/sh\nsystemctl reload nginx\n' >"$RENEW_HOOK"
    chmod 0755 "$RENEW_HOOK"
    activate "$HERE/$DOMAIN.conf"
    echo "HTTPS active. Check: curl -I https://$DOMAIN/"
    ;;
*)
    echo "usage: sudo $0 http|tls" >&2
    exit 2
    ;;
esac
