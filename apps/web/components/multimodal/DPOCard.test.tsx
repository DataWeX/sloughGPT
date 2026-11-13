import { describe, it, expect } from 'vitest'
import { default as DPOCardDefault, DPOCard } from './DPOCard'

describe('DPOCard re-export', () => {
  it('named export is defined', () => {
    expect(DPOCard).toBeDefined()
  })

  it('default export is defined', () => {
    expect(DPOCardDefault).toBeDefined()
  })
})
