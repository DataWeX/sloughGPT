import { describe, expect, it } from 'vitest'

import { routeMatchesPath } from './route-match'

describe('routeMatchesPath', () => {
  it('matches with or without trailing slash', () => {
    expect(routeMatchesPath('/chat/', '/chat')).toBe(true)
    expect(routeMatchesPath('/chat', '/chat')).toBe(true)
    expect(routeMatchesPath('/chat/', '/chat/')).toBe(true)
  })

  it('matches root', () => {
    expect(routeMatchesPath('/', '/')).toBe(true)
    expect(routeMatchesPath('', '/')).toBe(true)
  })

  it('does not match different segments', () => {
    expect(routeMatchesPath('/chat/', '/models')).toBe(false)
    expect(routeMatchesPath('/chat/foo', '/chat')).toBe(false)
  })
})
