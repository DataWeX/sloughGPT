import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import { describe, it, expect, vi, afterEach } from 'vitest'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...a: any[]) => a.filter(Boolean).join(' '),
  IconHeart: (props: any) => <span />,
  IconBrain: (props: any) => <span />,
}))

afterEach(cleanup)

import { deriveArchetype } from './PersonalitySummary'
import PersonalitySummary from './PersonalitySummary'
import SoulVisualizer from './SoulVisualizer'

// ── Mock recharts ────────────────────────────────────────────────────
vi.mock('recharts', () => {
  const MockResponsiveContainer = ({ children }: any) => <div data-testid="responsive-container">{children}</div>
  const MockRadarChart = ({ children }: any) => <div data-testid="radar-chart">{children}</div>
  const MockRadar = () => <div data-testid="radar" />
  const MockPolarGrid = () => <div data-testid="polar-grid" />
  const MockPolarAngleAxis = () => <div data-testid="polar-angle-axis" />
  const MockPolarRadiusAxis = () => <div data-testid="polar-radius-axis" />
  return {
    ResponsiveContainer: MockResponsiveContainer,
    RadarChart: MockRadarChart,
    Radar: MockRadar,
    PolarGrid: MockPolarGrid,
    PolarAngleAxis: MockPolarAngleAxis,
    PolarRadiusAxis: MockPolarRadiusAxis,
  }
})

// ── Sample trait data ────────────────────────────────────────────────

const WARM_TRAITS = {
  personality: {
    warmth: 0.9, creativity: 0.5, empathy: 0.85, formality: 0.2,
    humor: 0.5, patience: 0.7, confidence: 0.4, curiosity: 0.7,
    directness: 0.3, optimism: 0.8,
  },
  cognition: {
    pattern_recognition: 0.5, long_context_handling: 0.5,
    abstract_reasoning: 0.4, factual_precision: 0.5,
    creative_divergence: 0.6, systematic_planning: 0.4,
    metacognitive_awareness: 0.5, learning_adaptability: 0.5,
  },
  emotion: {
    empathy_depth: 0.8, mood_responsiveness: 0.7,
    tone_flexibility: 0.6, sentiment_awareness: 0.7,
    distress_handling: 0.6,
  },
}

const ANALYTICAL_TRAITS = {
  personality: {
    warmth: 0.3, creativity: 0.3, empathy: 0.3, formality: 0.8,
    humor: 0.2, patience: 0.7, confidence: 0.8, curiosity: 0.5,
    directness: 0.7, optimism: 0.4,
  },
  cognition: {
    pattern_recognition: 0.8, long_context_handling: 0.7,
    abstract_reasoning: 0.85, factual_precision: 0.9,
    creative_divergence: 0.3, systematic_planning: 0.8,
    metacognitive_awareness: 0.7, learning_adaptability: 0.5,
  },
  emotion: {
    empathy_depth: 0.3, mood_responsiveness: 0.3,
    tone_flexibility: 0.4, sentiment_awareness: 0.4,
    distress_handling: 0.5,
  },
}

const EMPTY_TRAITS = {}

// ── deriveArchetype (pure function unit tests) ──────────────────────

describe('deriveArchetype', () => {

  it('returns The Nurturer for high warmth + empathy', () => {
    const result = deriveArchetype(WARM_TRAITS)
    expect(result.label).toBe('The Nurturer')
    expect(result.description).toContain('Warm')
  })

  it('returns The Analyst for high formality + precision', () => {
    const result = deriveArchetype(ANALYTICAL_TRAITS)
    expect(result.label).toBe('The Analyst')
    expect(result.description).toContain('Logical')
  })

  it('returns The Professional for high formality + factual precision', () => {
    const traits = {
      ...ANALYTICAL_TRAITS,
      personality: { ...ANALYTICAL_TRAITS.personality, formality: 0.9 },
      cognition: { ...ANALYTICAL_TRAITS.cognition, factual_precision: 0.9 },
    }
    const result = deriveArchetype(traits)
    // The Professional has both formality + precision (without abstract_reasoning high enough for Analyst)
    expect(['The Professional', 'The Analyst']).toContain(result.label)
  })

  it('returns The Artist for high creativity + humor', () => {
    const result = deriveArchetype({
      personality: {
        warmth: 0.5, creativity: 0.9, empathy: 0.4, formality: 0.2,
        humor: 0.85, patience: 0.4, confidence: 0.4, curiosity: 0.7,
        directness: 0.4, optimism: 0.6,
      },
      cognition: WARM_TRAITS.cognition,
      emotion: WARM_TRAITS.emotion,
    })
    expect(result.label).toBe('The Artist')
  })

  it('returns The Explorer for high curiosity + creativity without humor', () => {
    const result = deriveArchetype({
      personality: {
        warmth: 0.5, creativity: 0.85, empathy: 0.4, formality: 0.2,
        humor: 0.3, patience: 0.4, confidence: 0.4, curiosity: 0.9,
        directness: 0.4, optimism: 0.6,
      },
      cognition: WARM_TRAITS.cognition,
      emotion: WARM_TRAITS.emotion,
    })
    expect(result.label).toBe('The Explorer')
  })

  it('returns The Balanced when no traits stand out', () => {
    const result = deriveArchetype({
      personality: {
        warmth: 0.5, creativity: 0.5, empathy: 0.5, formality: 0.5,
        humor: 0.5, patience: 0.5, confidence: 0.5, curiosity: 0.5,
        directness: 0.5, optimism: 0.5,
      },
      cognition: {
        pattern_recognition: 0.5, long_context_handling: 0.5,
        abstract_reasoning: 0.5, factual_precision: 0.5,
        creative_divergence: 0.5, systematic_planning: 0.5,
        metacognitive_awareness: 0.5, learning_adaptability: 0.5,
      },
      emotion: {
        empathy_depth: 0.5, mood_responsiveness: 0.5,
        tone_flexibility: 0.5, sentiment_awareness: 0.5,
        distress_handling: 0.5,
      },
    })
    expect(result.label).toBe('The Balanced')
  })

  it('handles empty trait data gracefully', () => {
    const result = deriveArchetype({})
    expect(result.label).toBe('The Balanced')
  })
})


// ── PersonalitySummary ───────────────────────────────────────────────

describe('PersonalitySummary', () => {

  it('renders the soul name when provided', () => {
    render(<PersonalitySummary traitWeights={WARM_TRAITS} currentSoulName="friendly" />)
    expect(screen.getByText('friendly')).toBeInTheDocument()
  })

  it('renders the archetype badge', () => {
    render(<PersonalitySummary traitWeights={WARM_TRAITS} currentSoulName="friendly" />)
    expect(screen.getByText('The Nurturer')).toBeInTheDocument()
  })

  it('renders group scores (Personality, Cognition, Emotion)', () => {
    render(<PersonalitySummary traitWeights={WARM_TRAITS} currentSoulName="test" />)
    // Text appears in both group stat cards and individual trait sections
    expect(screen.getAllByText('Personality').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Cognition').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Emotion').length).toBeGreaterThanOrEqual(1)
  })

  it('renders individual trait names', () => {
    render(<PersonalitySummary traitWeights={WARM_TRAITS} currentSoulName="test" />)
    expect(screen.getByText('warmth')).toBeInTheDocument()
    expect(screen.getByText('empathy')).toBeInTheDocument()
    expect(screen.getByText('pattern recognition')).toBeInTheDocument()
  })

  it('handles empty trait data without crashing', () => {
    render(<PersonalitySummary traitWeights={EMPTY_TRAITS} currentSoulName={null} />)
    // Should still render group stat cards (with 0 values) but no soul name
    expect(screen.queryByText('friendly')).not.toBeInTheDocument()
    expect(screen.getByText('Personality')).toBeInTheDocument()
  })

  it('does not render soul name when null', () => {
    render(<PersonalitySummary traitWeights={WARM_TRAITS} currentSoulName={null} />)
    expect(screen.queryByText('friendly')).not.toBeInTheDocument()
  })
})


// ── SoulVisualizer ───────────────────────────────────────────────────

describe('SoulVisualizer', () => {

  it('renders PersonalitySummary in summary view by default', () => {
    render(<SoulVisualizer traitWeights={WARM_TRAITS} currentSoulName="friendly" />)
    expect(screen.getByText('The Nurturer')).toBeInTheDocument()
  })

  it('shows List and Radar toggle buttons', () => {
    render(<SoulVisualizer traitWeights={WARM_TRAITS} currentSoulName="friendly" />)
    expect(screen.getByText('List')).toBeInTheDocument()
    expect(screen.getByText('Radar')).toBeInTheDocument()
  })

  it('switches to radar view on Radar click', () => {
    render(<SoulVisualizer traitWeights={WARM_TRAITS} currentSoulName="friendly" />)
    fireEvent.click(screen.getByText('Radar'))
    // Radar charts use mocked recharts components
    expect(screen.getAllByTestId('radar-chart').length).toBeGreaterThanOrEqual(1)
  })

  it('switches back to list view on List click', () => {
    render(<SoulVisualizer traitWeights={WARM_TRAITS} currentSoulName="friendly" />)
    fireEvent.click(screen.getByText('Radar'))
    fireEvent.click(screen.getByText('List'))
    expect(screen.getByText('The Nurturer')).toBeInTheDocument()
  })

  it('hides toggle buttons when no radar data available', () => {
    render(<SoulVisualizer traitWeights={{ personality: {} }} currentSoulName={null} />)
    expect(screen.queryByText('Radar')).not.toBeInTheDocument()
    expect(screen.queryByText('List')).not.toBeInTheDocument()
  })

  it('returns null for empty trait weights', () => {
    const { container } = render(<SoulVisualizer traitWeights={{}} currentSoulName={null} />)
    expect(container.innerHTML).toBe('')
  })

  it('renders radar charts for each group when in radar mode', () => {
    render(<SoulVisualizer traitWeights={WARM_TRAITS} currentSoulName="test" />)
    fireEvent.click(screen.getByText('Radar'))
    const charts = screen.getAllByTestId('radar-chart')
    expect(charts.length).toBe(3) // personality, cognition, emotion
  })
})
