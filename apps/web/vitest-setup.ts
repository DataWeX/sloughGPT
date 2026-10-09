import '@testing-library/jest-dom/vitest'
import { cleanup } from '@testing-library/react'
import { createElement } from 'react'
import { afterEach, vi } from 'vitest'

// react-router Link needs a Router context; unit tests render leaves without one.
vi.mock('@/vite/next-compat/link', () => ({
  default: ({
    href,
    children,
    className,
    ...rest
  }: {
    href?: unknown
    children?: unknown
    className?: string
    [key: string]: unknown
  }) => {
    const to = typeof href === 'string' ? href : String(href ?? '#')
    const { replace: _r, scroll: _s, prefetch: _p, passHref: _ph, ...anchor } = rest as Record<
      string,
      unknown
    >
    return createElement('a', { href: to, className, ...anchor }, children as never)
  },
}))

afterEach(() => {
  cleanup()
})

if (typeof window !== 'undefined') {
  class ResizeObserverMock {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
  const win = window as any
  win.ResizeObserver = ResizeObserverMock
}

// Tests must never ship logs to the real backend: suites that don't mock
// @/lib/dev-log (e.g. useChatMode.test.ts) fire real trackEvents, which
// LogTransport batches into POST /errors/logs/ingest and floods the server
// output buffer. Neutralize that endpoint only — other fetches pass through.
// Individual tests that stub fetch via vi.stubGlobal bypass this wrapper.
const _originalFetch = globalThis.fetch?.bind(globalThis)
if (_originalFetch) {
  globalThis.fetch = ((input: RequestInfo | URL, init?: RequestInit) => {
    const url =
      typeof input === 'string'
        ? input
        : input instanceof URL
          ? input.href
          : (input as Request).url
    if (url.includes('/errors/logs/ingest')) {
      return Promise.resolve(
        new Response(JSON.stringify({ ok: true }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }),
      )
    }
    return _originalFetch(input, init)
  }) as typeof fetch
}
