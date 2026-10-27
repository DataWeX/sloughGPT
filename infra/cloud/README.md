# Free cloud deployment targets

## Frontend — Cloudflare Pages
- Cost: free
- Pros: unlimited bandwidth, global CDN, Next.js supported
- Cons: no long-running backend; only static/SSR output
- Config: `infra/cloud/cloudflare-pages.toml`

## Backend — Render.com
- Cost: free web service
- Pros: Docker support, public URL, HTTPS
- Cons: 512MB RAM limit; spin-down after inactivity
- Config: `infra/cloud/render.yaml`

## Backend — Fly.io
- Cost: free tier (3 shared VMs)
- Pros: real VM, Docker, persistent IP option
- Cons: 256MB RAM per VM; model must be very small/quantized
- Config: `infra/cloud/fly.toml`

## Git-sync CD — GitHub Actions + Tailscale + Cloudflare Tunnel
- Cost: Tailscale (free for 100 devices) + Cloudflare (free)
- Pros: zero-downtime deploys on every git push; no open firewall ports;
  Cloudflare CDN + SSL termination at the edge; full control of the server
- Cons: requires a dedicated VM (not a free tier)
- Config: `.github/workflows/deploy.yml` + `infra/scripts/deploy-web.sh`

### How it works

```
  git push → GitHub Actions
                │
                ├── build:  npm ci → npm run build:vite  →  dist-vite/
                │
                └── deploy:  Tailscale GitHub Action
                              connects runner to your tailnet
                              │
                              ├── rsync dist-vite/  →  server:/var/www/html/
                              ├── ssh → reload nginx + health check
                              └── purge Cloudflare CDN cache
```

Cloudflare Tunnel (`cloudflared`) runs as a systemd service on the server and
proxies traffic from Cloudflare's edge to nginx. No firewall ports need to be
opened — the server initiates an outbound TLS connection to Cloudflare.

### Server setup (one-time)

```bash
# 1. Install Tailscale and join the tailnet
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up --authkey=TS_KEY_FROM_ADMIN_CONSOLE --hostname=deploy-vm

# 2. Install nginx
sudo apt-get install -y nginx
sudo cp ../docker/nginx/nginx.conf /etc/nginx/sites-available/default
# (adjust the API proxy_pass to point at your API server)
sudo systemctl enable --now nginx

# 3. Install cloudflared and create a tunnel
#    https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /usr/local/bin/cloudflared
chmod +x /usr/local/bin/cloudflared
cloudflared tunnel create  # → copies credentials to ~/.cloudflared/
cloudflared tunnel route dns my-tunnel example.com  # → Cloudflare DNS
sudo cloudflared service install
sudo systemctl enable --now cloudflared
```

### GitHub secrets / variables

| Name | Scope | Required | Description |
|---|---|---|---|
| `TS_AUTHKEY` | secret | yes | Tailscale pre-auth key with tag `tag:ci` |
| `DEPLOY_HOST` | secret | yes | Tailscale IP of the server (e.g. `100.x.x.x`) |
| `DEPLOY_SSH_KEY` | secret | yes | SSH private key (public key in server `authorized_keys`) |
| `DEPLOY_USER` | secret | no | SSH user (default: `root`) |
| `CLOUDFLARE_API_TOKEN` | secret | no | Token with Zone:Cache:Edit scope |
| `CF_ZONE_ID` | variable | no | Cloudflare zone ID (for cache purge) |

## Notes
- The default model is ~2.5GB unquantized. Free tiers cannot hold that.
- Use a smaller model or 4-bit quantization for free hosting.
- API and Web must be on the same domain to avoid CORS.
