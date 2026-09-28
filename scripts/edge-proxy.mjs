import http from 'node:http'
// Edge socket relay — browser/harness → :8080 (here) → slough-gateway → core.
// Ported from ~/.cache/opencode-cdp/edge-proxy.mjs (repo-owned now).
//
// This process is a dumb byte pipe: CORS preflight answers, header override,
// forward, pipe back. All policy (compression, health contract, path filters,
// timeouts) lives in the Rust gateway behind it — TARGET_PORT points there.
const TARGET = {
  host: '127.0.0.1',
  port: Number(process.env.TARGET_PORT || 8081),
}
const EDGE_PORT = Number(process.env.EDGE_PORT || 8080)
const CORS = {
  'access-control-allow-origin': '*',
  'access-control-allow-methods': 'GET,POST,PUT,PATCH,DELETE,OPTIONS',
  'access-control-allow-headers': '*',
  'access-control-expose-headers': '*',
}
http
  .createServer((req, res) => {
    if (req.method === 'OPTIONS') {
      res.writeHead(204, CORS)
      res.end()
      return
    }
    const p = http.request(
      {
        ...TARGET,
        path: req.url,
        method: req.method,
        headers: { ...req.headers, host: `${TARGET.host}:${TARGET.port}` },
      },
      (pr) => {
        res.writeHead(pr.statusCode, { ...pr.headers, ...CORS })
        pr.pipe(res)
      },
    )
    p.on('error', (e) => {
      res.writeHead(502, { 'content-type': 'text/plain', ...CORS })
      res.end('proxy err ' + e.message)
    })
    req.pipe(p)
  })
  .listen({ port: EDGE_PORT, host: '::', ipv6Only: false }, () =>
    console.log(`edge-proxy ${EDGE_PORT} -> ${TARGET.port} (dual-stack + CORS)`),
  )
