import {describe, it, expect} from 'vitest'
import {PUBLIC_API_URL} from './config'

// Same-origin contract (card 20261123_135845): the frontend never names a
// backend port. The server owns the port — dev relays through vite
// (vite/api-relay-plugin), the served site through the gateway — so the
// browser sees exactly one origin. The old contract (absolute
// http://localhost:8000) is retired: user `define` keys do not apply under
// vite 8 dev, so `process.env.NEXT_PUBLIC_API_URL` reached the browser raw
// and threw at import, killing http-client's whole chain (chat included)
// while vitest stayed green in Node where `process` exists.
describe('config', () => {
  it('PUBLIC_API_URL is same-origin (empty base)', () => {
    expect(PUBLIC_API_URL).toBe('')
  })

  it('never names a backend origin or port', () => {
    expect(typeof PUBLIC_API_URL).toBe('string')
    expect(PUBLIC_API_URL).not.toMatch(/^https?:/)
    expect(PUBLIC_API_URL).not.toContain(':')
  })

  it('concatenates to root-relative paths (the consumer pattern)', () => {
    expect(`${PUBLIC_API_URL}/chat`).toBe('/chat')
    expect(`${PUBLIC_API_URL}/info`).toBe('/info')
    expect(`${PUBLIC_API_URL}/chat/audio/s1/m1`).toBe('/chat/audio/s1/m1')
  })
})
