import { useTheme, useThemeName } from 'tamagui'

import { MOBILE_COLORS } from './palette.generated'

/**
 * Noir Violet Design System — Mobile Color Access
 *
 * Every value comes from palette.generated.ts, which is generated from
 * packages/strui/tokens/palette.json — the single source shared with the web CSS
 * variables in globals.css. Never hand-edit values here or in the generated file;
 * edit palette.json and run `node packages/strui/scripts/gen-tokens.mjs`.
 *
 * Use this hook in components that need runtime color access.
 *
 * For static values, prefer Tamagui theme tokens ($background, $color, etc.).
 */

type ColorKey = keyof typeof MOBILE_COLORS

/** '#RRGGBB' -> 'rgba(R, G, B, opacity)' — alpha helpers derive from palette values. */
function alpha(hexColor: string, opacity: number): string {
  const n = Number.parseInt(hexColor.slice(1), 16)
  const r = (n >> 16) & 255
  const g = (n >> 8) & 255
  const b = n & 255
  return `rgba(${r}, ${g}, ${b}, ${opacity})`
}

export function useColors() {
  const theme = useTheme()
  const themeName = useThemeName()
  const isDark = themeName === 'dark'

  const v = (key: ColorKey): string => (isDark ? MOBILE_COLORS[key].dark : MOBILE_COLORS[key].light)

  return {
    // ── Core ──────────────────────────────────────────
    primary: v('primary'),
    primaryLight: v('primaryLight'),
    accent: v('accent'),

    // ── Backgrounds ───────────────────────────────────
    background: v('background'),
    backgroundHover: v('backgroundHover'),
    backgroundPress: v('backgroundPress'),
    card: v('card'),
    popover: v('popover'),
    chatBg: v('chatBg'),

    // ── Text ──────────────────────────────────────────
    text: v('text'),
    textMuted: v('textMuted'),
    textSecondary: v('textSecondary'),
    textOnPrimary: v('textOnPrimary'),

    // ── Borders ───────────────────────────────────────
    border: v('border'),
    borderHover: v('borderHover'),

    // ── Secondary / Muted ─────────────────────────────
    secondary: v('secondary'),
    secondaryForeground: v('secondaryForeground'),
    muted: v('muted'),
    mutedForeground: v('mutedForeground'),

    // ── Semantic ──────────────────────────────────────
    error: v('error'),
    errorDark: v('errorDark'),
    errorLight: v('errorLight'),
    success: v('success'),
    successDark: v('successDark'),
    successLight: v('successLight'),
    warning: v('warning'),
    warningDark: v('warningDark'),
    warningLight: v('warningLight'),
    info: v('info'),
    infoDark: v('infoDark'),
    infoLight: v('infoLight'),
    white: v('white'),

    // ── Alpha Utilities ───────────────────────────────
    overlay: (opacity: number) => `rgba(0, 0, 0, ${opacity})`,
    primaryAlpha: (opacity: number) => alpha(v('primary'), opacity),
    errorAlpha: (opacity: number) => alpha(v('error'), opacity),
    errorDarkAlpha: (opacity: number) => alpha(v('errorDark'), opacity),
    successAlpha: (opacity: number) => alpha(v('success'), opacity),
    warningAlpha: (opacity: number) => alpha(v('warning'), opacity),
    whiteAlpha: (opacity: number) => alpha(v('white'), opacity),
  }
}
