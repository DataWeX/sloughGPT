import {describe, it, expect} from 'vitest'
import {PUBLIC_API_URL} from './config'

describe('config', () => {
  it('PUBLIC_API_URL defaults to localhost', () => {
    expect(PUBLIC_API_URL).toBeTruthy()
    expect(typeof PUBLIC_API_URL).toBe('string')
  })

  it('PUBLIC_API_URL is a valid URL', () => {
    expect(PUBLIC_API_URL).toMatch(/^https?:\/\//)
  })

  it('PUBLIC_API_URL ends with no trailing slash', () => {
    expect(PUBLIC_API_URL.endsWith('/')).toBe(false)
  })

  it('PUBLIC_API_URL contains port 8000', () => {
    expect(PUBLIC_API_URL).toContain('8000')
  })

  it('PUBLIC_API_URL is exported as a constant', () => {
    expect(typeof PUBLIC_API_URL).toBe('string')
    expect(PUBLIC_API_URL.length).toBeGreaterThan(0)
  })
})
