import { describe, expect, it } from 'vitest'

import { whatsNewItems } from './whats-new-data'

describe('whatsNewItems', () => {
  it('is a non-empty array', () => {
    expect(Array.isArray(whatsNewItems)).toBe(true)
    expect(whatsNewItems.length).toBeGreaterThan(0)
  })

  it('each item has required fields', () => {
    for (const item of whatsNewItems) {
      expect(item.id).toBeTruthy()
      expect(item.title).toBeTruthy()
      expect(item.description).toBeTruthy()
      expect(item.icon).toBeTruthy()
      expect(item.date).toMatch(/^\d{4}-\d{2}-\d{2}$/)
    }
  })
})
