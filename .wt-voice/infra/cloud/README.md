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

## Notes
- The default model is ~2.5GB unquantized. Free tiers cannot hold that.
- Use a smaller model or 4-bit quantization for free hosting.
- API and Web must be on the same domain to avoid CORS.
