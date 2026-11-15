/**
 * Personality page data model: the profile/persona/conflict/history
 * shapes and the display label maps. Pure data — no React, no IO.
 */
export interface SavedPersona {
  id: string
  name: string
  values: string[]
  /** Trait weights — present on personas saved/listed by the consciousness API. */
  traits?: Record<string, number>
  /** The traits that define this persona (furthest from neutral). */
  top_traits?: Array<{ key: string; value: number }>
  /** True when this persona's profile is the current live profile. */
  active?: boolean
}

export interface PersonalityConflict {
  type: string
  severity: string
  message: string
  fields: string[]
}

export interface PersonalityProfile {
  values: string[]
  goals: string[]
  voice: Record<string, number>
  style: Record<string, boolean>
  traits: Record<string, number>
  interests: string[]
  avoid: string[]
}

export interface PersonalityHistory {
  timestamp: number
  voice: Record<string, number>
  traits: Record<string, number>
}

export const VOICE_LABELS: Record<string, string> = {
  formality: 'Formality',
  warmth: 'Warmth',
  confidence: 'Confidence',
  humor: 'Humor',
  verbosity: 'Verbosity',
  empathy: 'Empathy',
}

export const TRAIT_LABELS: Record<string, string> = {
  openness: 'Openness',
  conscientiousness: 'Conscientiousness',
  extraversion: 'Extraversion',
  agreeableness: 'Agreeableness',
  neuroticism: 'Emotional Stability',
}

export const STYLE_LABELS: Record<string, string> = {
  use_examples: 'Use Examples',
  ask_follow_ups: 'Ask Follow-ups',
  acknowledge_uncertainty: 'Acknowledge Uncertainty',
  use_analogies: 'Use Analogies',
  break_down_complex_topics: 'Break Down Complex Topics',
}
