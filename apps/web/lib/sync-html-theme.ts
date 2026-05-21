import { THEME_IDS, type StoredThemeId, type ThemeMode } from '@/lib/theme-storage'

/** Apply mode + accent classes on ``<html>`` without wiping Next/font classes. */
export function syncHtmlTheme(mode: ThemeMode, theme: StoredThemeId) {
  if (typeof document === 'undefined') return
  const root = document.documentElement
  root.classList.remove('light', 'dark')
  root.classList.add(mode)
  THEME_IDS.forEach((id) => root.classList.remove(`theme-${id}`))
  root.classList.add(`theme-${theme}`)
}
