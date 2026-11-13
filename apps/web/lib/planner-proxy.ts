import { NextResponse } from '@/vite/next-compat/server'

/**
 * Base URL of the local planner backend (`planner gui`, served by
 * packages/planner/src/planner/gui.py on 127.0.0.1:8787 by default).
 * Override with PLANNER_API_BASE when the planner runs elsewhere.
 */
const PLANNER_API_BASE = process.env.PLANNER_API_BASE ?? 'http://127.0.0.1:8787'

export const PLANNER_URL = PLANNER_API_BASE

/**
 * First-party relay marker (card 20261108_142919). Every call site must
 * pass this header in `PlannerProxyInit.headers`; `proxyRequest` consumes
 * it and strips it before forwarding. A proxy call without it is refused
 * with 403 — the gate that makes the planner proxy a first-party-only
 * seam instead of an open fetch relay.
 */
const RELAY_HEADER = 'x-slo-relay'
const RELAY_VALUE = '1'

export interface PlannerProxyInit {
  method?: string
  body?: string
  headers?: Record<string, string>
}

function hasRelayHeader(init?: PlannerProxyInit): boolean {
  return Object.entries(init?.headers ?? {}).some(
    ([k, v]) => k.toLowerCase() === RELAY_HEADER && v === RELAY_VALUE,
  )
}

/** App headers to forward — the relay marker never leaves this process. */
function forwardHeaders(init?: PlannerProxyInit): Record<string, string> | undefined {
  const out: Record<string, string> = {}
  for (const [k, v] of Object.entries(init?.headers ?? {})) {
    if (k.toLowerCase() !== RELAY_HEADER) out[k] = v
  }
  return Object.keys(out).length > 0 ? out : undefined
}

/**
 * Forward a request to the planner backend. The `path` already carries any
 * query string the route assembled. Returns the backend's status, body, and
 * content-type, or a 502 when the planner backend is unreachable.
 *
 * Two gates before any fetch (card 20261108_142919):
 *   1. first-party relay — the caller must set `x-slo-relay: 1`, proving the
 *      call comes from a server route of this app, not from request data;
 *   2. host pin — the resolved destination must have exactly the host of
 *      PLANNER_API_BASE (closes `@evil.com` userinfo escapes when `path`
 *      ever becomes caller-influenced).
 */
export async function proxyRequest(path: string, init?: PlannerProxyInit): Promise<NextResponse> {
  if (!hasRelayHeader(init)) {
    return NextResponse.json(
      { error: 'Planner relay requires the first-party X-Slo-Relay header' },
      { status: 403 },
    )
  }

  const url = `${PLANNER_API_BASE}${path}`
  let target: URL
  try {
    target = new URL(url)
  } catch {
    return NextResponse.json(
      { error: 'Planner relay destination is not a valid URL' },
      { status: 403 },
    )
  }
  if (target.host !== new URL(PLANNER_API_BASE).host) {
    return NextResponse.json({ error: 'Planner relay host pin violated' }, { status: 403 })
  }

  let res: Response
  try {
    res = await fetch(url, {
      method: init?.method ?? 'GET',
      body: init?.body,
      headers: forwardHeaders(init) as HeadersInit | undefined,
    })
  } catch {
    return NextResponse.json(
      { error: `Planner backend unreachable at ${PLANNER_API_BASE}` },
      { status: 502 },
    )
  }

  const text = await res.text()
  const contentType = res.headers.get('content-type') ?? 'text/plain'
  return new NextResponse(text, {
    status: res.status,
    headers: { 'content-type': contentType },
  })
}
