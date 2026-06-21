import { describe, expect, it } from 'vitest'
import { isStoredThemeId, THEME_IDS } from '../theme-storage'

describe('isStoredThemeId', () => {
  it('returns true for valid theme IDs', () => {
    for (const id of THEME_IDS) {
      expect(isStoredThemeId(id)).toBe(true)
    }
  })

  it('returns false for invalid theme IDs', () => {
    expect(isStoredThemeId('yellow')).toBe(false)
    expect(isStoredThemeId('')).toBe(false)
    expect(isStoredThemeId('blueish')).toBe(false)
  })

  it('returns false for null', () => {
    expect(isStoredThemeId(null)).toBe(false)
  })

  it('returns false for undefined', () => {
    expect(isStoredThemeId(undefined as unknown as string)).toBe(false)
  })

  it('THEME_IDS contains expected values', () => {
    expect(THEME_IDS).toEqual(['blue', 'purple', 'pink', 'red', 'orange', 'green', 'teal'])
  })
})
