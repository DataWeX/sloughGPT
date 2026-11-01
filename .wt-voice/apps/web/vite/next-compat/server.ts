// next/server value shim for Vite SSR / middleware (dev only).
// Route handlers only need NextResponse.json + a Request-compatible type.

export type NextRequest = Request

export class NextResponse extends Response {
  static json(body: unknown, init?: ResponseInit): Response {
    const headers = new Headers(init?.headers)
    if (!headers.has('content-type')) {
      headers.set('content-type', 'application/json')
    }
    return new Response(JSON.stringify(body), { ...init, headers })
  }

  static next(init?: ResponseInit): Response {
    return new Response(null, { status: 200, ...init })
  }

  static redirect(url: string | URL, status = 307): Response {
    return new Response(null, {
      status,
      headers: { location: String(url) },
    })
  }
}
