/** Keys synced with ``ThemeProvider`` and the inline theme bootstrap in ``app/layout.tsx``. */

export const THEME_STORAGE_KEY = 'sloughgpt_theme'
export const MODE_STORAGE_KEY = 'sloughgpt_mode'

export const THEME_IDS = ['blue', 'purple', 'pink', 'red', 'orange', 'green', 'teal'] as const

export type StoredThemeId = (typeof THEME_IDS)[number]

export type ThemeMode = 'dark' | 'light'

export function isStoredThemeId(value: string | null): value is StoredThemeId {
  return value != null && (THEME_IDS as readonly string[]).includes(value)
}
