/** Keys synced with ``ThemeProvider`` and the inline theme bootstrap in ``app/layout.tsx``. */

import { OS_BLUE_ID, OS_BLUE_LABEL, OS_BLUE_SWATCH } from './os-blue-palette'
import { ROSE_BLUE_ID, ROSE_BLUE_LABEL, ROSE_BLUE_SWATCH } from './rose-blue-palette'

export const THEME_STORAGE_KEY = 'man_theme'
export const MODE_STORAGE_KEY = 'man_mode'
export const PALETTE_STORAGE_KEY = 'man_palette'

export const THEME_IDS = ['blue', 'purple', 'pink', 'red', 'orange', 'green', 'teal'] as const

export type StoredThemeId = (typeof THEME_IDS)[number]

/** The canvases the Appearance picker offers.
 *  ``high-contrast`` is a layer ON the dark canvas — see ``globals.css``
 *  ``html.dark.high-contrast``, which both ``syncHtmlTheme`` and the
 *  ``index.html`` bootstrap apply alongside ``dark``. */
export const MODE_IDS = ['dark', 'light', 'high-contrast'] as const

export type ThemeMode = (typeof MODE_IDS)[number]

export function isThemeMode(value: string | null): value is ThemeMode {
  return value != null && (MODE_IDS as readonly string[]).includes(value)
}

/** True when the canvas is dark — ``high-contrast`` sits on the dark canvas. */
export function isDarkMode(mode: ThemeMode): boolean {
  return mode !== 'light'
}

/** Target of the binary light/dark toggle: leaving high contrast returns to light. */
export function toggleMode(mode: ThemeMode): ThemeMode {
  return mode === 'light' ? 'dark' : 'light'
}

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

/**
 * Appearance-picker tile previews.
 *
 * The picker shows every mode's colours at once while only one mode's tokens
 * are live, so a tile cannot reference vars — the same data exemption as
 * ``THEMES``/``PALETTE_COLORS`` (design-criteria HEX_BASELINE). Values mirror
 * ``globals.css``: ``:root`` for light, ``html.dark`` for dark, and pure
 * black/white with a yellow accent for high contrast.
 */
export interface ThemePreview {
  id: ThemeMode
  label: string
  /** Canvas */
  bg: string
  /** Body text */
  fg: string
  /** Raised surface (card) */
  panel: string
  /** Dividers and muted rules */
  border: string
  /** Primary action fill */
  primary: string
  /** Text on the primary fill */
  primaryFg: string
}

/** Tile order matches the shipped picker: Dark, Light, High Contrast. */
export const THEME_PREVIEWS: ThemePreview[] = [
  {
    id: 'dark',
    label: 'Dark',
    bg: '#110f18',
    fg: '#eeeaf8',
    panel: '#1c1926',
    border: '#342e48',
    primary: '#c0aaf4',
    primaryFg: '#191624',
  },
  {
    id: 'light',
    label: 'Light',
    bg: '#f8f6fc',
    fg: '#191624',
    panel: '#ffffff',
    border: '#e4e0f2',
    primary: '#7c52c4',
    primaryFg: '#faf8ff',
  },
  {
    id: 'high-contrast',
    label: 'High Contrast',
    bg: '#000000',
    fg: '#ffffff',
    panel: '#000000',
    border: '#ffffff',
    primary: '#ffff00',
    primaryFg: '#000000',
  },
]
