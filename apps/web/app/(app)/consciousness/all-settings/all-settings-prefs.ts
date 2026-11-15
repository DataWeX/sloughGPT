'use client'

/**
 * Preference model for All Settings: the level labels, the localStorage
 * persistence for display/data/notification prefs and personality, the
 * status shapes, and the personality presets. Pure data + storage — no
 * React, no network.
 */
export const LEVEL_LABELS = ['Off', 'Basic', 'Full', 'Deep']
export const LEVEL_DESCRIPTIONS = [
  'Consciousness system disabled',
  'Basic qualia and self-reflection',
  'Full self-model with belief tracking',
  'Deep reflection with narrative generation',
]

export const ALL_PREFS_KEY = 'consciousness_all_settings'
export const PERSONALITY_KEY = 'consciousness_personality'
export const NOTIFICATIONS_KEY = 'consciousness_notifications_prefs'

export interface AllPrefs {
  showInChat: boolean
  showInStatusBar: boolean
  realTimeUpdates: boolean
  autoReflect: boolean
  autoEvolve: boolean
  autoRefresh: boolean
  browserNotifications: boolean
  soundNotifications: boolean
  alertThresholdLow: number
  alertThresholdHigh: number
}

export interface PersonalityPrefs {
  warmth: number
  confidence: number
  humor: number
  empathy: number
  openness: number
  agreeableness: number
  conscientiousness: number
  summary: string
}

export interface ConfigStatus {
  level: number
  enabled: boolean
}

export interface FullStatus {
  enabled: boolean
  level: number
  episodes: number
  beliefs: Record<string, number>
  training: {
    is_training: boolean
    total_pairs: number
  }
}

export const DEFAULT_PREFS: AllPrefs = {
  showInChat: true,
  showInStatusBar: true,
  realTimeUpdates: false,
  autoReflect: false,
  autoEvolve: false,
  autoRefresh: false,
  browserNotifications: false,
  soundNotifications: false,
  alertThresholdLow: 30,
  alertThresholdHigh: 80,
}

export const DEFAULT_PERSONALITY: PersonalityPrefs = {
  warmth: 50,
  confidence: 50,
  humor: 50,
  empathy: 50,
  openness: 50,
  agreeableness: 50,
  conscientiousness: 50,
  summary: 'Default personality',
}

export function loadAllPrefs(): AllPrefs {
  if (typeof window === 'undefined') return DEFAULT_PREFS
  try {
    const raw = localStorage.getItem(ALL_PREFS_KEY)
    if (!raw) return DEFAULT_PREFS
    return { ...DEFAULT_PREFS, ...JSON.parse(raw) }
  } catch {
    return DEFAULT_PREFS
  }
}

export function saveAllPrefs(prefs: AllPrefs) {
  try {
    localStorage.setItem(ALL_PREFS_KEY, JSON.stringify(prefs))
  } catch {
    /* ignored */
  }
}

export function loadPersonality(): PersonalityPrefs {
  if (typeof window === 'undefined') return DEFAULT_PERSONALITY
  try {
    const raw = localStorage.getItem(PERSONALITY_KEY)
    if (!raw) return DEFAULT_PERSONALITY
    return { ...DEFAULT_PERSONALITY, ...JSON.parse(raw) }
  } catch {
    return DEFAULT_PERSONALITY
  }
}

export function savePersonality(prefs: PersonalityPrefs) {
  try {
    localStorage.setItem(PERSONALITY_KEY, JSON.stringify(prefs))
  } catch {
    /* ignored */
  }
}

export const PRESETS: Record<string, Partial<PersonalityPrefs>> = {
  default: DEFAULT_PERSONALITY,
  formal: {
    warmth: 30,
    confidence: 70,
    humor: 10,
    empathy: 40,
    openness: 40,
    agreeableness: 60,
    conscientiousness: 80,
  },
  creative: {
    warmth: 60,
    confidence: 50,
    humor: 70,
    empathy: 50,
    openness: 90,
    agreeableness: 50,
    conscientiousness: 30,
  },
  analyst: {
    warmth: 20,
    confidence: 80,
    humor: 20,
    empathy: 30,
    openness: 60,
    agreeableness: 40,
    conscientiousness: 90,
  },
  empathetic: {
    warmth: 90,
    confidence: 40,
    humor: 30,
    empathy: 90,
    openness: 70,
    agreeableness: 80,
    conscientiousness: 50,
  },
  minimal: {
    warmth: 20,
    confidence: 60,
    humor: 10,
    empathy: 20,
    openness: 30,
    agreeableness: 40,
    conscientiousness: 60,
  },
}
