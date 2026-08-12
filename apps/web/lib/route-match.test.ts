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
    expect(routeMatchesPath('/channels', '/chat')).toBe(false)
  })

  it('prefix-matches nested routes to parent', () => {
    expect(routeMatchesPath('/chat/foo', '/chat')).toBe(true)
    expect(routeMatchesPath('/training/job/xyz', '/training')).toBe(true)
    expect(routeMatchesPath('/models/detail/gpt2', '/models')).toBe(true)
  })

  it('does not match partial segment names', () => {
    expect(routeMatchesPath('/chatt', '/chat')).toBe(false)
    expect(routeMatchesPath('/model', '/models')).toBe(false)
  })
})
