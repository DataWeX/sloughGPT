import {useTheme, useThemeName} from 'tamagui';

/**
 * Noir Violet Design System — Mobile Color Access
 *
 * All values match the web CSS variables defined in globals.css.
 * Use this hook in components that need runtime color access.
 *
 * For static values, prefer Tamagui theme tokens ($background, $color, etc.).
 */
export function useColors() {
  const theme = useTheme();
  const themeName = useThemeName();
  const isDark = themeName === 'dark';

  return {
    // ── Core ──────────────────────────────────────────
    primary: isDark ? '#C0AAF4' : '#7C52C4',
    primaryLight: isDark ? '#C0AAF4' : '#7C52C4',
    accent: isDark ? '#F0B082' : '#EC915F',

    // ── Backgrounds ───────────────────────────────────
    background: isDark ? '#110F18' : '#F8F6FC',
    backgroundHover: isDark ? '#1C1926' : '#F0EDFA',
    backgroundPress: isDark ? '#262234' : '#E6E2F4',
    card: isDark ? '#1C1926' : '#FFFFFF',
    popover: isDark ? '#201D2C' : '#FFFFFF',
    chatBg: isDark ? '#16141E' : '#F6F2ED',

    // ── Text ──────────────────────────────────────────
    text: isDark ? '#EEEAF8' : '#191624',
    textMuted: isDark ? '#968CAC' : '#827A96',
    textSecondary: isDark ? '#EEEAF8' : '#2A2537',
    textOnPrimary: isDark ? '#191624' : '#FAF8FF',

    // ── Borders ───────────────────────────────────────
    border: isDark ? '#342E48' : '#E4E0F2',
    borderHover: isDark ? '#C0AAF4' : '#7C52C4',

    // ── Secondary / Muted ─────────────────────────────
    secondary: isDark ? '#322C44' : '#EDE8F8',
    secondaryForeground: isDark ? '#EEEAF8' : '#2A2537',
    muted: isDark ? '#262234' : '#F4F2F8',
    mutedForeground: isDark ? '#968CAC' : '#827A96',

    // ── Semantic ──────────────────────────────────────
    error: isDark ? '#EB646E' : '#DC505A',
    errorDark: '#D44C56',
    errorLight: isDark ? '#3B1A1A' : '#FDE8E8',
    success: isDark ? '#48C08C' : '#34B07D',
    successDark: '#2E9B7C',
    successLight: isDark ? '#1A2E22' : '#E8F5EE',
    warning: isDark ? '#F0C050' : '#ECA83C',
    warningDark: '#E8A83C',
    warningLight: isDark ? '#2E2410' : '#FFF8E7',
    info: isDark ? '#78AFF0' : '#5A96DC',
    infoDark: '#2563EB',
    infoLight: isDark ? '#1A2240' : '#EFF6FF',
    white: '#FFFFFF',

    // ── Alpha Utilities ───────────────────────────────
    overlay: (opacity: number) => `rgba(0, 0, 0, ${opacity})`,
    primaryAlpha: (opacity: number) =>
      isDark ? `rgba(192, 170, 244, ${opacity})` : `rgba(124, 82, 196, ${opacity})`,
    errorAlpha: (opacity: number) =>
      isDark ? `rgba(235, 100, 110, ${opacity})` : `rgba(220, 80, 90, ${opacity})`,
    errorDarkAlpha: (opacity: number) => `rgba(212, 76, 86, ${opacity})`,
    successAlpha: (opacity: number) =>
      isDark ? `rgba(72, 192, 140, ${opacity})` : `rgba(52, 176, 125, ${opacity})`,
    warningAlpha: (opacity: number) =>
      isDark ? `rgba(240, 192, 80, ${opacity})` : `rgba(236, 168, 60, ${opacity})`,
    whiteAlpha: (opacity: number) => `rgba(255, 255, 255, ${opacity})`,
  };
}
