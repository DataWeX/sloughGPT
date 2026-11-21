import {
  THEME_IDS,
  PALETTE_IDS,
  MODE_IDS,
  type StoredThemeId,
  type ThemeMode,
  type StoredPaletteId,
} from '@/lib/theme-storage'

/** Apply mode + accent + palette classes on ``<html>`` without wiping Next/font classes. */
export function syncHtmlTheme(
  mode: ThemeMode,
  theme: StoredThemeId,
  palette?: StoredPaletteId,
): void {
  if (typeof document === 'undefined') return
  const root = document.documentElement
  MODE_IDS.forEach((id) => root.classList.remove(id))
  // High contrast layers ON the dark canvas: both classes land so every
  // `html.dark` rule still applies and `html.dark.high-contrast` overrides it.
  root.classList.add(...(mode === 'high-contrast' ? ['dark', 'high-contrast'] : [mode]))
  THEME_IDS.forEach((id) => root.classList.remove(`theme-${id}`))
  root.classList.add(`theme-${theme}`)
  PALETTE_IDS.forEach((id) => root.classList.remove(`palette-${id}`))
  if (palette && palette !== 'noir-violet') {
    root.classList.add(`palette-${palette}`)
  }
}
