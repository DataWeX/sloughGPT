import { describe, expect, it } from 'vitest'
import { THEME_IDS, isStoredThemeId } from './theme-storage'

describe('THEME_IDS', () => {
  it('contains 7 theme ids', () => {
    expect(THEME_IDS).toEqual(['blue', 'purple', 'pink', 'red', 'orange', 'green', 'teal'])
  })
})

describe('isStoredThemeId', () => {
  it('returns true for valid theme id', () => {
    expect(isStoredThemeId('blue')).toBe(true)
    expect(isStoredThemeId('purple')).toBe(true)
    expect(isStoredThemeId('teal')).toBe(true)
  })

  it('returns false for invalid theme id', () => {
    expect(isStoredThemeId('yellow')).toBe(false)
    expect(isStoredThemeId('')).toBe(false)
    expect(isStoredThemeId('dark')).toBe(false)
    expect(isStoredThemeId('BLUE')).toBe(false)
  })

  it('returns false for null', () => {
    expect(isStoredThemeId(null)).toBe(false)
  })

  it('returns false for undefined', () => {
    expect(isStoredThemeId(undefined as unknown as string | null)).toBe(false)
  })

  it('returns true for every valid theme id', () => {
    for (const id of THEME_IDS) {
      expect(isStoredThemeId(id)).toBe(true)
    }
  })

  it('has exactly 7 themes', () => {
    expect(THEME_IDS).toHaveLength(7)
  })
})
