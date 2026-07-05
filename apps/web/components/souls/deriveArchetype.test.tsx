import { describe, it, expect } from 'vitest'
import { deriveArchetype } from './PersonalitySummary'

const weights = (p: Record<string, number> = {}, c: Record<string, number> = {}, e: Record<string, number> = {}) => ({
  personality: {
    warmth: 0.5, creativity: 0.5, empathy: 0.5, formality: 0.5,
    humor: 0.5, patience: 0.5, confidence: 0.5, curiosity: 0.5,
    directness: 0.5, optimism: 0.5,
    ...p,
  },
  cognition: {
    pattern_recognition: 0.5, long_context_handling: 0.5,
    abstract_reasoning: 0.5, factual_precision: 0.5,
    creative_divergence: 0.5, systematic_planning: 0.5,
    metacognitive_awareness: 0.5, learning_adaptability: 0.5,
    ...c,
  },
  emotion: {
    empathy_depth: 0.5, mood_responsiveness: 0.5,
    tone_flexibility: 0.5, sentiment_awareness: 0.5,
    distress_handling: 0.5,
    ...e,
  },
})

describe('deriveArchetype', () => {
  it('returns Balanced for default weights', () => {
    expect(deriveArchetype(weights()).label).toBe('The Balanced')
  })

  it('returns Nurturer when warmth, empathy, empathy_depth are high', () => {
    const result = deriveArchetype(weights({ warmth: 0.8, empathy: 0.8 }, {}, { empathy_depth: 0.8 }))
    expect(result.label).toBe('The Nurturer')
  })

  it('returns Analyst when abstract_reasoning and factual_precision are high', () => {
    const result = deriveArchetype(weights({}, { abstract_reasoning: 0.8, factual_precision: 0.8 }))
    expect(result.label).toBe('The Analyst')
  })

  it('returns Artist when creativity and humor are high', () => {
    const result = deriveArchetype(weights({ creativity: 0.9, humor: 0.8 }))
    expect(result.label).toBe('The Artist')
  })

  it('returns Explorer when curiosity and creativity are high', () => {
    const result = deriveArchetype(weights({ curiosity: 0.9, creativity: 0.8 }))
    expect(result.label).toBe('The Explorer')
  })

  it('returns Leader when confidence and directness are high', () => {
    const result = deriveArchetype(weights({ confidence: 0.9, directness: 0.8 }))
    expect(result.label).toBe('The Leader')
  })

  it('returns Companion when humor and warmth are high', () => {
    const result = deriveArchetype(weights({ humor: 0.9, warmth: 0.8 }))
    expect(result.label).toBe('The Companion')
  })

  it('returns Listener when empathy and patience are high', () => {
    const result = deriveArchetype(weights({ empathy: 0.9, patience: 0.8 }))
    expect(result.label).toBe('The Listener')
  })

  it('returns Optimist when optimism is high', () => {
    const result = deriveArchetype(weights({ optimism: 0.9 }))
    expect(result.label).toBe('The Optimist')
  })

  it('returns Professional when formality and factual_precision are high', () => {
    const result = deriveArchetype(weights({ formality: 0.9 }, { factual_precision: 0.8 }))
    expect(result.label).toBe('The Professional')
  })

  it('prefers highest scoring archetype when multiple match', () => {
    // Nurturer: warmth(0.9)+empathy(0.9)+empathy_depth(0.9) = 2.7
    // Listener: empathy(0.9)+patience(0.9) = 1.8
    // Nurturer should win with higher sum
    const result = deriveArchetype(weights(
      { warmth: 0.9, empathy: 0.9, patience: 0.9 },
      {},
      { empathy_depth: 0.9 }
    ))
    expect(result.label).toBe('The Nurturer')
  })

  it('handles missing groups gracefully', () => {
    const result = deriveArchetype({ personality: {}, cognition: {}, emotion: {} })
    expect(result.label).toBe('The Balanced')
  })
})
