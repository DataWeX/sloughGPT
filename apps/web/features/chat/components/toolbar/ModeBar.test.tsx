import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import { ModeBar } from './ModeBar'

describe('ModeBar', () => {
  afterEach(() => {
    cleanup()
  })

  const defaultProps = {
    mode: 'chat' as const,
    tone: 'Friendly',
    type: 'Email',
    decideStructure: 'Pros & Cons',
    difficulty: 'Simple',
    langPair: 'EN→ES',
    brainstormTopic: 'Name Ideas',
    wellnessType: 'Sleep Story',
    createStyle: 'Realistic',
    rewriteStyle: 'Fix Grammar',
    onModeChange: vi.fn(),
    onToneChange: vi.fn(),
    onTypeChange: vi.fn(),
    onRewriteStyleChange: vi.fn(),
    onDecideStructureChange: vi.fn(),
    onDifficultyChange: vi.fn(),
    onLangPairChange: vi.fn(),
    onBrainstormTopicChange: vi.fn(),
    onWellnessTypeChange: vi.fn(),
    onCreateStyleChange: vi.fn(),
  }

  it('renders mode trigger with current mode label', () => {
    const { container } = render(<ModeBar {...defaultProps} />)
    expect(screen.getByText('Chat')).toBeInTheDocument()
    const svgs = container.querySelectorAll('svg')
    expect(svgs.length).toBeGreaterThanOrEqual(1)
  })

  it('shows sub-options for write mode', () => {
    render(<ModeBar {...defaultProps} mode="write" />)
    expect(screen.getByText('Tone')).toBeInTheDocument()
    expect(screen.getByText('Friendly')).toBeInTheDocument()
    expect(screen.getByText('Email')).toBeInTheDocument()
  })

  it('calls onToneChange when a tone pill is clicked', () => {
    const onToneChange = vi.fn()
    render(<ModeBar {...defaultProps} mode="write" onToneChange={onToneChange} />)
    fireEvent.click(screen.getByText('Funny'))
    expect(onToneChange).toHaveBeenCalledWith('Funny')
  })

  it('calls onTypeChange when a type pill is clicked', () => {
    const onTypeChange = vi.fn()
    render(<ModeBar {...defaultProps} mode="write" onTypeChange={onTypeChange} />)
    fireEvent.click(screen.getByText('Poem'))
    expect(onTypeChange).toHaveBeenCalledWith('Poem')
  })

  it('highlights the selected tone', () => {
    render(<ModeBar {...defaultProps} mode="write" tone="Professional" />)
    expect(screen.getByText('Professional')).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByText('Friendly')).toHaveAttribute('aria-pressed', 'false')
  })

  it('shows decide sub-options in decide mode', () => {
    render(<ModeBar {...defaultProps} mode="decide" />)
    expect(screen.getByText('Output')).toBeInTheDocument()
    expect(screen.getByText('Pros & Cons')).toBeInTheDocument()
    expect(screen.getByText('Comparison')).toBeInTheDocument()
    expect(screen.getByText('Deep Analysis')).toBeInTheDocument()
  })

  it('calls onDecideStructureChange when clicked', () => {
    const onDecideStructureChange = vi.fn()
    render(<ModeBar {...defaultProps} mode="decide" onDecideStructureChange={onDecideStructureChange} />)
    fireEvent.click(screen.getByText('Comparison'))
    expect(onDecideStructureChange).toHaveBeenCalledWith('Comparison')
  })

  it('highlights selected decide structure', () => {
    render(<ModeBar {...defaultProps} mode="decide" decideStructure="Deep Analysis" />)
    expect(screen.getByText('Deep Analysis')).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByText('Pros & Cons')).toHaveAttribute('aria-pressed', 'false')
  })

  it('hides write sub-options in chat mode', () => {
    render(<ModeBar {...defaultProps} mode="chat" />)
    expect(screen.queryByText('Tone')).not.toBeInTheDocument()
    expect(screen.queryByText('Friendly')).not.toBeInTheDocument()
  })

  it('hides decide sub-options in write mode', () => {
    render(<ModeBar {...defaultProps} mode="write" />)
    expect(screen.queryByText('Pros & Cons')).not.toBeInTheDocument()
  })

  it('hides write sub-options in decide mode', () => {
    render(<ModeBar {...defaultProps} mode="decide" />)
    expect(screen.queryByText('Friendly')).not.toBeInTheDocument()
  })

  it('shows explain difficulty sub-options', () => {
    render(<ModeBar {...defaultProps} mode="explain" />)
    expect(screen.getByText('Level')).toBeInTheDocument()
    expect(screen.getByText('Simple')).toBeInTheDocument()
    expect(screen.getByText('Moderate')).toBeInTheDocument()
    expect(screen.getByText('Expert')).toBeInTheDocument()
  })

  it('calls onDifficultyChange when clicked', () => {
    const onDifficultyChange = vi.fn()
    render(<ModeBar {...defaultProps} mode="explain" onDifficultyChange={onDifficultyChange} />)
    fireEvent.click(screen.getByText('Expert'))
    expect(onDifficultyChange).toHaveBeenCalledWith('Expert')
  })

  it('shows translate language pair sub-options', () => {
    render(<ModeBar {...defaultProps} mode="translate" />)
    expect(screen.getByText('To')).toBeInTheDocument()
    expect(screen.getByText('EN→ES')).toBeInTheDocument()
    expect(screen.getByText('EN→FR')).toBeInTheDocument()
    expect(screen.getByText('EN→DE')).toBeInTheDocument()
  })

  it('calls onLangPairChange when clicked', () => {
    const onLangPairChange = vi.fn()
    render(<ModeBar {...defaultProps} mode="translate" onLangPairChange={onLangPairChange} />)
    fireEvent.click(screen.getByText('EN→FR'))
    expect(onLangPairChange).toHaveBeenCalledWith('EN→FR')
  })

  it('hides explain sub-options in chat mode', () => {
    render(<ModeBar {...defaultProps} mode="chat" />)
    expect(screen.queryByText('Moderate')).not.toBeInTheDocument()
  })

  it('hides translate sub-options in write mode', () => {
    render(<ModeBar {...defaultProps} mode="write" />)
    expect(screen.queryByText('EN→ES')).not.toBeInTheDocument()
  })

  it('shows brainstorm topic sub-options', () => {
    render(<ModeBar {...defaultProps} mode="brainstorm" />)
    expect(screen.getByText('Topic')).toBeInTheDocument()
    expect(screen.getByText('Name Ideas')).toBeInTheDocument()
    expect(screen.getByText('Gift Ideas')).toBeInTheDocument()
    expect(screen.getByText('Plan an Event')).toBeInTheDocument()
  })

  it('calls onBrainstormTopicChange when clicked', () => {
    const onBrainstormTopicChange = vi.fn()
    render(<ModeBar {...defaultProps} mode="brainstorm" onBrainstormTopicChange={onBrainstormTopicChange} />)
    fireEvent.click(screen.getByText('Weekend Plans'))
    expect(onBrainstormTopicChange).toHaveBeenCalledWith('Weekend Plans')
  })

  it('hides brainstorm sub-options in chat mode', () => {
    render(<ModeBar {...defaultProps} mode="chat" />)
    expect(screen.queryByText('Name Ideas')).not.toBeInTheDocument()
    expect(screen.queryByText('Gift Ideas')).not.toBeInTheDocument()
  })

  it('shows wellness type sub-options', () => {
    render(<ModeBar {...defaultProps} mode="wellness" />)
    expect(screen.getByText('Type')).toBeInTheDocument()
    expect(screen.getByText('Sleep Story')).toBeInTheDocument()
    expect(screen.getByText('Meditation')).toBeInTheDocument()
  })

  it('calls onWellnessTypeChange when clicked', () => {
    const onWellnessTypeChange = vi.fn()
    render(<ModeBar {...defaultProps} mode="wellness" onWellnessTypeChange={onWellnessTypeChange} />)
    fireEvent.click(screen.getByText('Breathing'))
    expect(onWellnessTypeChange).toHaveBeenCalledWith('Breathing')
  })

  it('hides wellness sub-options in chat mode', () => {
    render(<ModeBar {...defaultProps} mode="chat" />)
    expect(screen.queryByText('Sleep Story')).not.toBeInTheDocument()
    expect(screen.queryByText('Meditation')).not.toBeInTheDocument()
  })

  it('shows create style sub-options', () => {
    render(<ModeBar {...defaultProps} mode="create" />)
    expect(screen.getByText('Style')).toBeInTheDocument()
    expect(screen.getByText('Realistic')).toBeInTheDocument()
    expect(screen.getByText('Cartoon')).toBeInTheDocument()
    expect(screen.getByText('Watercolor')).toBeInTheDocument()
    expect(screen.getByText('Sketch')).toBeInTheDocument()
    expect(screen.getByText('Fantasy')).toBeInTheDocument()
  })

  it('calls onCreateStyleChange when clicked', () => {
    const onCreateStyleChange = vi.fn()
    render(<ModeBar {...defaultProps} mode="create" onCreateStyleChange={onCreateStyleChange} />)
    fireEvent.click(screen.getByText('Watercolor'))
    expect(onCreateStyleChange).toHaveBeenCalledWith('Watercolor')
  })

  it('highlights selected style', () => {
    render(<ModeBar {...defaultProps} mode="create" createStyle="Fantasy" />)
    expect(screen.getByText('Fantasy')).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByText('Realistic')).toHaveAttribute('aria-pressed', 'false')
  })

  it('hides create sub-options in chat mode', () => {
    render(<ModeBar {...defaultProps} mode="chat" />)
    expect(screen.queryByText('Realistic')).not.toBeInTheDocument()
    expect(screen.queryByText('Cartoon')).not.toBeInTheDocument()
  })

  it('updates trigger label when mode changes', () => {
    const { rerender } = render(<ModeBar {...defaultProps} mode="chat" />)
    expect(screen.getByText('Chat')).toBeInTheDocument()
    rerender(<ModeBar {...defaultProps} mode="write" />)
    expect(screen.getByText('Write')).toBeInTheDocument()
  })
})
