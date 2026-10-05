/**
 * Shared design scales for apps/mobile — spacing (S), radius (R), type (T).
 *
 * Source of truth: docs/design/DESIGN_SYSTEM.md § Mobile (React Native / Tamagui)
 * — Font Mapping, Type Scale (RN), Spacing Grid, Radius, No Magic Numbers.
 *
 * Hard rule (No Magic Numbers): components must not inline raw spacing,
 * radius, or fontSize values — import S/R/T from here instead.
 * Literal exceptions: icon/glyph sizes (14/16/18/20) and 1–2px hairlines/borders.
 * Colors are exempt — they come from theme tokens or useColors() only.
 *
 * When in doubt, match the web class first (the table is web → RN), then the
 * semantic Use column in the Radius table.
 */
// scales.ts — MessageBubble.tsx's local constants, promoted to one shared module
export const S = {xs: 4, sm: 8, md: 12, lg: 16, xl: 20, xxl: 24, xxxl: 32} as const // 4pt grid
export const R = {xs: 2, sm: 6, md: 10, lg: 14, xl: 16, full: 9999} as const // radius

export const T = { // type styles
  page: {fontSize: 24, lineHeight: 30, fontWeight: '600'},
  section: {fontSize: 16, lineHeight: 22, fontWeight: '500'},
  body: {fontSize: 15, lineHeight: 22},
  caption: {fontSize: 12, lineHeight: 16},
  label: {fontSize: 12, lineHeight: 16, letterSpacing: 0.6},
  badge: {fontSize: 11, lineHeight: 15, fontWeight: '500'},
} as const
