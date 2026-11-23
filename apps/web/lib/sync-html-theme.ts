import {
  THEME_IDS,
  PALETTE_IDS,
  MODE_IDS,
  type StoredThemeId,
  type ThemeMode,
  type StoredPaletteId,
} from '@/lib/theme-storage'

/**
 * Class armed for the duration of a colour swap so every token consumer eases
 * over the same 200ms (see `html.theme-transitioning` in globals.css).
 *
 * Applied to `<html>` BEFORE the mode/theme/palette groups are rewritten, so
 * the browser styles the class change and the token change in one frame and
 * the transition actually runs; removed once the swap has been committed.
 */
const TRANSITION_CLASS = 'theme-transitioning'

/** Longest transition the scoped rule declares — disarm after it elapses. */
const TRANSITION_MS = 200

/** Set once the first (mount) sync has run: never animate initial paint. */
let hasSynced = false

/** Test seam — resets the mount guard so suites start from a cold document. */
export function resetThemeSyncGuard(): void {
  hasSynced = false
}

/**
 * Apply mode + accent + palette classes on ``<html>`` without wiping Next/font classes.
 *
 * @param animate - ease the swap instead of snapping. Callers pass this only
 *   for user-initiated changes; the mount sync must never animate, or the
 *   whole page fades in from nothing on every load. Defaults to `true` once
 *   a first sync has happened, so every change path (settings tiles,
 *   ThemeSwitcher, CommandPalette, Ctrl+Shift+D) animates without each one
 *   having to remember to ask.
 */
export function syncHtmlTheme(
  mode: ThemeMode,
  theme: StoredThemeId,
  palette?: StoredPaletteId,
  animate: boolean = hasSynced,
): void {
  if (typeof document === 'undefined') return
  const root = document.documentElement

  // Reduced motion is honoured in CSS (the scoped rule collapses to
  // `transition: none`), so arming here is still correct — it just won't move.
  const shouldAnimate = animate && hasSynced
  if (shouldAnimate) {
    root.classList.add(TRANSITION_CLASS)
    // Force the armed style to be computed before the token groups change,
    // otherwise both land in the same style recalc and nothing transitions.
    void root.offsetWidth
  }

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

  if (shouldAnimate) {
    // Disarm on the next macrotask so the browser has committed the class
    // change with the transition armed; a rAF pair would still be inside the
    // same frame and drop it.
    setTimeout(() => root.classList.remove(TRANSITION_CLASS), TRANSITION_MS)
  }

  hasSynced = true
}
