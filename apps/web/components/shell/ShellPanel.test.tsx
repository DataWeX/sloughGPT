import { describe, it, expect } from 'vitest'
import { ShellPanel } from './ShellPanel'

describe('ShellPanel re-export', () => {
  it('exports a valid component', () => {
    expect(ShellPanel).toBeDefined()
    expect(typeof ShellPanel).toBe('function')
  })
})
