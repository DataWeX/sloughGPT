export const FONT_IDS = ['inter', 'outfit', 'space-grotesk', 'plus-jakarta'] as const

export type FontId = (typeof FONT_IDS)[number]

export const FONT_LABELS: Record<FontId, string> = {
  inter: 'Inter',
  outfit: 'Outfit',
  'space-grotesk': 'Space Grotesk',
  'plus-jakarta': 'Plus Jakarta Sans',
}

export const FONT_VARIABLES: Record<FontId, string> = {
  inter: '--font-inter',
  outfit: '--font-outfit',
  'space-grotesk': '--font-space-grotesk',
  'plus-jakarta': '--font-plus-jakarta',
}

export const FONT_CSS_CLASSES: Record<FontId, string> = {
  inter: 'font-body-inter',
  outfit: 'font-body-outfit',
  'space-grotesk': 'font-body-space-grotesk',
  'plus-jakarta': 'font-body-plus-jakarta',
}

export function isFontId(value: string | null): value is FontId {
  return value != null && (FONT_IDS as readonly string[]).includes(value)
}
