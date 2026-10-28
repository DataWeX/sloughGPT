/**
 * Operating System Blue — palette protocol.
 *
 * This module is the NORMATIVE source for the third opt-in palette (id
 * `system-cobalt`). Everything else is a projection of it:
 *
 *   - the `html.palette-system-cobalt` / `html.dark.palette-system-cobalt`
 *     blocks in `app/globals.css` (static, drift-guarded by
 *     `os-blue-palette.test.ts` — the test re-runs this derivation and
 *     asserts byte-identical CSS),
 *   - the ThemeSwitcher / settings swatch (via `theme-storage.ts`),
 *   - the "Palette Specification" section on the design lab (`/magazine`).
 *
 * Protocol, in three layers:
 *
 *   INPUTS   two base colors — light `#17599d` (the shade the user picked
 *            from the live preview, deliberately NOT the brighter #0067c0)
 *            and dark `#4cc2ff` (the Win11-style accent).
 *
 *   RULES    OKLCH hue-constant derivation: both ramps are generated at the
 *            base hue; chroma is scaled per shade (light desaturates toward
 *            the ends, dark scales with lightness) and the shade whose
 *            lightness target is nearest the base's own lightness SNAPS to
 *            the exact base color (the anchor). Surfaces are whisper tints
 *            of the same hue.
 *
 *   CONTRACT the state ladder — light darkens on interaction
 *            (selected → hover → active), dark lightens on hover and
 *            presses darker on active; "selected" IS the base color in
 *            both modes. Accessibility: white on light primary ≥ 4.5:1,
 *            dark primary-foreground on dark primary ≥ 4.5:1.
 */

export const OS_BLUE_ID = 'system-cobalt'
export const OS_BLUE_LABEL = 'Operating System Blue'
export const OS_BLUE_SWATCH = '#17599d'

export const OS_BLUE_BASES = {
  light: '#17599d',
  dark: '#4cc2ff',
} as const

/** Interaction-state shade per mode (shade index into the mode's ramp). */
export const OS_BLUE_STATES = {
  light: { selected: 700, hover: 800, active: 900, selectHoverBorder: 500 },
  dark: { selected: 800, hover: 900, active: 700, selectHoverBorder: 600 },
} as const

export type Mode = 'light' | 'dark'
export type Rgb = [number, number, number]
export type Shade = 50 | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900

export const SHADES: readonly Shade[] = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900]

// ── color math ────────────────────────────────────────────────────────────

export function hexToRgb(hex: string): Rgb {
  const h = hex.replace('#', '')
  return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)]
}

export function rgbToHex([r, g, b]: Rgb): string {
  const p = (n: number) => n.toString(16).padStart(2, '0')
  return `#${p(r)}${p(g)}${p(b)}`
}

function srgbToLinear(c: number): number {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4)
}

/** Linear light → sRGB transfer. Stays valid outside [0,1] (negatives map
    negative) so callers can detect out-of-gamut values before clamping. */
function linearToSrgb(c: number): number {
  return c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055
}

/** WCAG 2.1 relative luminance. */
function relativeLuminance([r, g, b]: Rgb): number {
  const R = srgbToLinear(r / 255)
  const G = srgbToLinear(g / 255)
  const B = srgbToLinear(b / 255)
  return 0.2126 * R + 0.7152 * G + 0.0722 * B
}

/** WCAG 2.1 contrast ratio (1..21). */
export function contrastRatio(a: Rgb, b: Rgb): number {
  const la = relativeLuminance(a)
  const lb = relativeLuminance(b)
  const [hi, lo] = la >= lb ? [la, lb] : [lb, la]
  return (hi + 0.05) / (lo + 0.05)
}

function rgbToOklch([r, g, b]: Rgb): { L: number; C: number; H: number } {
  const R = srgbToLinear(r / 255)
  const G = srgbToLinear(g / 255)
  const B = srgbToLinear(b / 255)
  const l = Math.cbrt(0.4122214708 * R + 0.5363325363 * G + 0.0514459929 * B)
  const m = Math.cbrt(0.2119034982 * R + 0.6806995451 * G + 0.1073969566 * B)
  const s = Math.cbrt(0.0883024619 * R + 0.2817188376 * G + 0.6299787005 * B)
  const L = 0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s
  const A = 1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s
  const Bb = 0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s
  return { L, C: Math.hypot(A, Bb), H: Math.atan2(Bb, A) }
}

/** Unclamped sRGB in 0..255 floats — out-of-gamut channels go <0 or >255. */
function oklchToRgbRaw(L: number, C: number, H: number): [number, number, number] {
  const A = C * Math.cos(H)
  const Bb = C * Math.sin(H)
  const l_ = L + 0.3963377774 * A + 0.2158037573 * Bb
  const m_ = L - 0.1055613458 * A - 0.0638541728 * Bb
  const s_ = L - 0.0894841775 * A - 1.291485548 * Bb
  const l = l_ ** 3
  const m = m_ ** 3
  const s = s_ ** 3
  return [
    linearToSrgb(4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s) * 255,
    linearToSrgb(-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s) * 255,
    linearToSrgb(-0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s) * 255,
  ]
}

function oklchToRgb(L: number, C: number, H: number): Rgb {
  return oklchToRgbRaw(L, C, H).map((v) => Math.round(Math.min(255, Math.max(0, v)))) as Rgb
}

/** Max chroma the sRGB gamut holds at (L, H). The gamut is convex, so
    in-gamut chroma is the interval [0, Cmax] — binary search finds Cmax.
    The protocol fits chroma DOWN to this instead of letting the encoder
    silently clamp: a clamp desaturates and drags hue, which would break
    the hue-constancy invariant at the pale/deep ends of the ramp. */
function maxInGamutChroma(L: number, H: number): number {
  let lo = 0
  let hi = 0.37 // above the max chroma sRGB represents anywhere
  for (let i = 0; i < 24; i++) {
    const mid = (lo + hi) / 2
    const [r, g, b] = oklchToRgbRaw(L, mid, H)
    const inGamut = r >= 0 && r <= 255 && g >= 0 && g <= 255 && b >= 0 && b <= 255
    if (inGamut) lo = mid
    else hi = mid
  }
  return lo
}

// ── protocol rules ────────────────────────────────────────────────────────

/** Lightness targets (OKLCH L) per shade, per mode. */
const LIGHTNESS: Record<Mode, Record<Shade, number>> = {
  light: {
    50: 0.976,
    100: 0.945,
    200: 0.885,
    300: 0.805,
    400: 0.71,
    500: 0.615,
    600: 0.545,
    700: 0.475,
    800: 0.405,
    900: 0.32,
  },
  dark: {
    50: 0.155,
    100: 0.205,
    200: 0.255,
    300: 0.305,
    400: 0.385,
    500: 0.475,
    600: 0.585,
    700: 0.7,
    800: 0.815,
    900: 0.9,
  },
}

/** Chroma scaling per shade — light peaks mid-ramp, dark grows with L. */
const CHROMA_SCALE: Record<Mode, (L: number) => number> = {
  light: (L) => Math.max(0.3, 1 - Math.abs(L - 0.55) * 1.25),
  dark: (L) => Math.max(0.055, Math.min(1, L * 1.55)),
}

export interface Ramp {
  shades: Record<Shade, Rgb>
  /** Shade whose lightness target is nearest the base's own — snapped to the exact base color. */
  anchor: Shade
}

export function deriveRamp(mode: Mode): Ramp {
  const base = hexToRgb(OS_BLUE_BASES[mode])
  const { L: baseL, C: baseC, H } = rgbToOklch(base)
  const shades = {} as Record<Shade, Rgb>
  for (const shade of SHADES) {
    const L = LIGHTNESS[mode][shade]
    // gamut-fit: at the pale/deep ends the target can leave sRGB; reduce
    // chroma (never let the encoder clamp behind the protocol's back)
    const C = Math.min(Math.max(0.004, baseC * CHROMA_SCALE[mode](L)), maxInGamutChroma(L, H))
    shades[shade] = oklchToRgb(L, C, H)
  }
  // anchor-snap: the shade nearest the base's own lightness IS the base color
  let anchor: Shade = SHADES[0]
  let best = Infinity
  for (const shade of SHADES) {
    const d = Math.abs(LIGHTNESS[mode][shade] - baseL)
    if (d < best) {
      best = d
      anchor = shade
    }
  }
  shades[anchor] = base
  return { shades, anchor }
}

// ── projections ───────────────────────────────────────────────────────────

const triplet = ([r, g, b]: Rgb) => `${r} ${g} ${b}`

/** Accent is a contrast color (house pattern: NP pairs a purple primary with a sky accent). */
const ACCENT: Record<Mode, string> = {
  light: '0 140 170',
  dark: '0 183 195',
}
const ACCENT_FG: Record<Mode, string> = {
  light: '244 253 255',
  dark: '6 18 22',
}

const DARK_FG = '235 241 248'
const LIGHT_PRIMARY_FG = '255 255 255'
const DARK_PRIMARY_FG = '8 18 28'

/**
 * Full palette token set for one mode — exactly the custom properties
 * written into the globals.css palette block (byte-identical; guarded).
 */
export function deriveTokens(mode: Mode): Record<string, string> {
  const { shades, anchor } = deriveRamp(mode)
  const primary = triplet(shades[anchor]) // selected IS the base color (the anchor shade)
  if (mode === 'light') {
    const surface = (L: number, cFrac: number) => {
      const { C, H } = rgbToOklch(hexToRgb(OS_BLUE_BASES.light))
      const chroma = Math.min(Math.max(0.002, C * cFrac), maxInGamutChroma(L, H))
      return oklchToRgb(L, chroma, H)
    }
    const border = triplet(surface(0.878, 0.32))
    const fg = triplet(surface(0.285, 0.16))
    return {
      background: triplet(surface(0.976, 0.1)),
      foreground: fg,
      card: triplet(surface(0.999, 0.05)),
      'card-foreground': fg,
      popover: triplet(surface(0.999, 0.05)),
      'popover-foreground': fg,
      primary,
      'primary-foreground': LIGHT_PRIMARY_FG,
      secondary: triplet(surface(0.933, 0.22)),
      'secondary-foreground': triplet(surface(0.3, 0.18)),
      muted: triplet(surface(0.951, 0.16)),
      'muted-foreground': triplet(surface(0.56, 0.3)),
      accent: ACCENT.light,
      'accent-foreground': ACCENT_FG.light,
      ring: primary,
      border,
      input: border,
      'chart-1': primary,
      'chart-2': ACCENT.light,
      'chart-3': triplet(shades[600]),
      'chart-4': '52 176 125',
      'chart-5': '220 80 90',
      'shadow-sm': '0 1px 2px rgba(35, 43, 52, 0.06)',
      'shadow-md': '0 4px 12px rgba(35, 43, 52, 0.08)',
      'shadow-lg': '0 8px 30px rgba(35, 43, 52, 0.10)',
      'shadow-xl': '0 16px 48px rgba(35, 43, 52, 0.14)',
    }
  }
  const border = triplet(shades[400])
  const darkPrimary = triplet(shades[800])
  return {
    'chat-bg': triplet(shades[50]),
    background: triplet(shades[50]),
    foreground: DARK_FG,
    card: triplet(shades[200]),
    'card-foreground': DARK_FG,
    popover: triplet(shades[300]),
    'popover-foreground': DARK_FG,
    primary: darkPrimary,
    'primary-foreground': DARK_PRIMARY_FG,
    secondary: triplet(shades[300]),
    'secondary-foreground': DARK_FG,
    muted: triplet(shades[200]),
    'muted-foreground': '152 163 178',
    accent: ACCENT.dark,
    'accent-foreground': ACCENT_FG.dark,
    ring: darkPrimary,
    border,
    input: border,
    'chart-1': darkPrimary,
    'chart-2': ACCENT.dark,
    'chart-3': triplet(shades[600]),
    'chart-4': '52 176 125',
    'chart-5': '235 100 110',
    'shadow-sm': '0 1px 2px rgba(0, 0, 0, 0.3)',
    'shadow-md': '0 4px 12px rgba(0, 0, 0, 0.4)',
    'shadow-lg': '0 8px 30px rgba(0, 0, 0, 0.5)',
    'shadow-xl': '0 16px 48px rgba(0, 0, 0, 0.6)',
  }
}
