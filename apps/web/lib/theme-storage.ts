/** Keys synced with ``ThemeProvider`` and the inline theme bootstrap in ``app/layout.tsx``. */

import { OS_BLUE_ID, OS_BLUE_LABEL, OS_BLUE_SWATCH } from './os-blue-palette'
import { ROSE_BLUE_ID, ROSE_BLUE_LABEL, ROSE_BLUE_SWATCH } from './rose-blue-palette'

export const THEME_STORAGE_KEY = 'man_theme'
export const MODE_STORAGE_KEY = 'man_mode'
export const PALETTE_STORAGE_KEY = 'man_palette'

export const THEME_IDS = ['blue', 'purple', 'pink', 'red', 'orange', 'green', 'teal'] as const

export type StoredThemeId = (typeof THEME_IDS)[number]

export type ThemeMode = 'dark' | 'light'

export const PALETTE_IDS = ['noir-violet', 'neural-precision', OS_BLUE_ID, ROSE_BLUE_ID] as const

export type StoredPaletteId = (typeof PALETTE_IDS)[number]

export const PALETTE_LABELS: Record<StoredPaletteId, string> = {
  'noir-violet': 'Noir Violet',
  'neural-precision': 'Neural Precision',
  [OS_BLUE_ID]: OS_BLUE_LABEL,
  [ROSE_BLUE_ID]: ROSE_BLUE_LABEL,
}

export const PALETTE_COLORS: Record<StoredPaletteId, string> = {
  'noir-violet': '#6339aa',
  'neural-precision': '#38bdf8',
  [OS_BLUE_ID]: OS_BLUE_SWATCH,
  [ROSE_BLUE_ID]: ROSE_BLUE_SWATCH,
}

export function isStoredThemeId(value: string | null): value is StoredThemeId {
  return value != null && (THEME_IDS as readonly string[]).includes(value)
}

export function isStoredPaletteId(value: string | null): value is StoredPaletteId {
  return value != null && (PALETTE_IDS as readonly string[]).includes(value)
}
