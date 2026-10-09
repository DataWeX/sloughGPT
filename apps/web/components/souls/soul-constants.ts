/** Shared soul trait group definitions — single source of truth. */

export const SOUL_GROUP_KEYS = ['personality', 'cognition', 'emotion'] as const
export type SoulGroupKey = (typeof SOUL_GROUP_KEYS)[number]

export const SOUL_GROUP_LABELS: Record<SoulGroupKey, string> = {
  personality: 'Personality',
  cognition: 'Cognition',
  emotion: 'Emotion',
}

export const SOUL_GROUP_COLORS: Record<SoulGroupKey, string> = {
  personality: 'rgb(var(--primary))',
  cognition: 'rgb(var(--chart-4))',
  emotion: 'rgb(var(--destructive))',
}

export const SOUL_GROUPS: { key: SoulGroupKey; label: string; color: string }[] =
  SOUL_GROUP_KEYS.map(key => ({
    key,
    label: SOUL_GROUP_LABELS[key],
    color: SOUL_GROUP_COLORS[key],
  }))
