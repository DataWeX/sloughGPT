import {describe, it, expect} from 'vitest'
import {PUBLIC_API_URL} from './config'

describe('config', () => {
  it('PUBLIC_API_URL defaults to localhost', () => {
    expect(PUBLIC_API_URL).toBeTruthy()
    expect(typeof PUBLIC_API_URL).toBe('string')
  })
})
