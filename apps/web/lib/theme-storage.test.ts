import { describe, expect, it } from 'vitest'
import { THEME_IDS, isStoredThemeId, PALETTE_IDS, PALETTE_LABELS, isStoredPaletteId } from './theme-storage'

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

describe('PALETTE_IDS', () => {
  it('contains 2 palette ids', () => {
    expect(PALETTE_IDS).toEqual(['noir-violet', 'neural-precision'])
  })

  it('has labels for every palette', () => {
    for (const id of PALETTE_IDS) {
      expect(PALETTE_LABELS[id]).toBeDefined()
      expect(typeof PALETTE_LABELS[id]).toBe('string')
    }
  })
})

describe('isStoredPaletteId', () => {
  it('returns true for valid palette id', () => {
    expect(isStoredPaletteId('noir-violet')).toBe(true)
    expect(isStoredPaletteId('neural-precision')).toBe(true)
  })

  it('returns false for invalid palette id', () => {
    expect(isStoredPaletteId('solarized')).toBe(false)
    expect(isStoredPaletteId('')).toBe(false)
    expect(isStoredPaletteId(null)).toBe(false)
  })
})
