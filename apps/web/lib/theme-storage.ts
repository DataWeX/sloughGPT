/** Keys synced with ``ThemeProvider`` and the inline theme bootstrap in ``app/layout.tsx``. */

export const THEME_STORAGE_KEY = 'man_theme'
export const MODE_STORAGE_KEY = 'man_mode'
export const PALETTE_STORAGE_KEY = 'man_palette'

export const THEME_IDS = ['blue', 'purple', 'pink', 'red', 'orange', 'green', 'teal'] as const

export type StoredThemeId = (typeof THEME_IDS)[number]

export type ThemeMode = 'dark' | 'light'

export const PALETTE_IDS = ['noir-violet', 'neural-precision'] as const

export type StoredPaletteId = (typeof PALETTE_IDS)[number]

export const PALETTE_LABELS: Record<StoredPaletteId, string> = {
  'noir-violet': 'Noir Violet',
  'neural-precision': 'Neural Precision',
}

export function isStoredThemeId(value: string | null): value is StoredThemeId {
  return value != null && (THEME_IDS as readonly string[]).includes(value)
}

export function isStoredPaletteId(value: string | null): value is StoredPaletteId {
  return value != null && (PALETTE_IDS as readonly string[]).includes(value)
}
