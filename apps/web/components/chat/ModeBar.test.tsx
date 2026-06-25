// @vitest-environment jsdom
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
    onModeChange: vi.fn(),
    onToneChange: vi.fn(),
    onTypeChange: vi.fn(),
    onDecideStructureChange: vi.fn(),
    onDifficultyChange: vi.fn(),
    onLangPairChange: vi.fn(),
    onBrainstormTopicChange: vi.fn(),
    onWellnessTypeChange: vi.fn(),
    onCreateStyleChange: vi.fn(),
  }

  it('renders all three mode buttons', () => {
    render(<ModeBar {...defaultProps} />)
    expect(screen.getAllByText('💬 Chat').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('✍️ Write').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('⚖️ Decide').length).toBeGreaterThanOrEqual(1)
  })

  it('shows Chat as active by default', () => {
    render(<ModeBar {...defaultProps} />)
    expect(screen.getAllByText('💬 Chat')[0]).toHaveAttribute('aria-pressed', 'true')
  })

  it('calls onModeChange when Write is clicked', () => {
    const onModeChange = vi.fn()
    render(<ModeBar {...defaultProps} onModeChange={onModeChange} />)
    fireEvent.click(screen.getAllByText('✍️ Write')[0])
    expect(onModeChange).toHaveBeenCalledWith('write')
  })

  it('shows tone chips in write mode', () => {
    render(<ModeBar {...defaultProps} mode="write" />)
    expect(screen.getAllByText('Friendly').length).toBeGreaterThanOrEqual(1)
  })

  it('shows type chips in write mode', () => {
    render(<ModeBar {...defaultProps} mode="write" />)
    expect(screen.getAllByText('Email').length).toBeGreaterThanOrEqual(1)
  })

  it('calls onToneChange when a tone chip is clicked', () => {
    const onToneChange = vi.fn()
    render(<ModeBar {...defaultProps} mode="write" onToneChange={onToneChange} />)
    fireEvent.click(screen.getAllByText('Funny')[0])
    expect(onToneChange).toHaveBeenCalledWith('Funny')
  })

  it('calls onTypeChange when a type chip is clicked', () => {
    const onTypeChange = vi.fn()
    render(<ModeBar {...defaultProps} mode="write" onTypeChange={onTypeChange} />)
    fireEvent.click(screen.getAllByText('Poem')[0])
    expect(onTypeChange).toHaveBeenCalledWith('Poem')
  })

  it('highlights the selected tone', () => {
    render(<ModeBar {...defaultProps} mode="write" tone="Professional" />)
    expect(screen.getAllByText('Professional')[0]).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getAllByText('Friendly')[0]).toHaveAttribute('aria-pressed', 'false')
  })

  it('shows decide structure chips in decide mode', () => {
    render(<ModeBar {...defaultProps} mode="decide" />)
    expect(screen.getAllByText('Pros & Cons').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Comparison').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Deep Analysis').length).toBeGreaterThanOrEqual(1)
  })

  it('calls onDecideStructureChange when clicked', () => {
    const onDecideStructureChange = vi.fn()
    render(<ModeBar {...defaultProps} mode="decide" onDecideStructureChange={onDecideStructureChange} />)
    fireEvent.click(screen.getAllByText('Comparison')[0])
    expect(onDecideStructureChange).toHaveBeenCalledWith('Comparison')
  })

  it('highlights selected decide structure', () => {
    render(<ModeBar {...defaultProps} mode="decide" decideStructure="Deep Analysis" />)
    expect(screen.getAllByText('Deep Analysis')[0]).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getAllByText('Pros & Cons')[0]).toHaveAttribute('aria-pressed', 'false')
  })

  it('hides write chips in chat mode', () => {
    render(<ModeBar {...defaultProps} mode="chat" />)
    expect(screen.queryAllByText('Friendly')).toHaveLength(0)
  })

  it('hides decide chips in write mode', () => {
    render(<ModeBar {...defaultProps} mode="write" />)
    expect(screen.queryAllByText('Pros & Cons')).toHaveLength(0)
  })

  it('hides write chips in decide mode', () => {
    render(<ModeBar {...defaultProps} mode="decide" />)
    expect(screen.queryAllByText('Friendly')).toHaveLength(0)
  })

  it('shows explain mode button and difficulty chips', () => {
    render(<ModeBar {...defaultProps} mode="explain" />)
    expect(screen.getAllByText('🔍 Explain').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Simple').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Moderate').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Expert').length).toBeGreaterThanOrEqual(1)
  })

  it('calls onDifficultyChange when a difficulty chip is clicked', () => {
    const onDifficultyChange = vi.fn()
    render(<ModeBar {...defaultProps} mode="explain" onDifficultyChange={onDifficultyChange} />)
    fireEvent.click(screen.getAllByText('Expert')[0])
    expect(onDifficultyChange).toHaveBeenCalledWith('Expert')
  })

  it('shows translate mode button and language pair chips', () => {
    render(<ModeBar {...defaultProps} mode="translate" />)
    expect(screen.getAllByText('🌐 Translate').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('EN→ES').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('EN→FR').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('EN→DE').length).toBeGreaterThanOrEqual(1)
  })

  it('calls onLangPairChange when a language chip is clicked', () => {
    const onLangPairChange = vi.fn()
    render(<ModeBar {...defaultProps} mode="translate" onLangPairChange={onLangPairChange} />)
    fireEvent.click(screen.getAllByText('EN→FR')[0])
    expect(onLangPairChange).toHaveBeenCalledWith('EN→FR')
  })

  it('hides explain chips in chat mode', () => {
    render(<ModeBar {...defaultProps} mode="chat" />)
    expect(screen.queryAllByText('Moderate')).toHaveLength(0)
  })

  it('hides translate chips in write mode', () => {
    render(<ModeBar {...defaultProps} mode="write" />)
    expect(screen.queryAllByText('EN→ES')).toHaveLength(0)
  })

  it('shows brainstorm mode button and topic chips', () => {
    render(<ModeBar {...defaultProps} mode="brainstorm" />)
    expect(screen.getAllByText('💡 Brainstorm').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Name Ideas').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Gift Ideas').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Plan an Event').length).toBeGreaterThanOrEqual(1)
  })

  it('calls onBrainstormTopicChange when a topic chip is clicked', () => {
    const onBrainstormTopicChange = vi.fn()
    render(<ModeBar {...defaultProps} mode="brainstorm" onBrainstormTopicChange={onBrainstormTopicChange} />)
    fireEvent.click(screen.getAllByText('Weekend Plans')[0])
    expect(onBrainstormTopicChange).toHaveBeenCalledWith('Weekend Plans')
  })

  it('hides brainstorm chips in chat mode', () => {
    render(<ModeBar {...defaultProps} mode="chat" />)
    expect(screen.queryAllByText('Name Ideas')).toHaveLength(0)
    expect(screen.queryAllByText('Gift Ideas')).toHaveLength(0)
  })

  it('shows wellness mode button and type chips', () => {
    render(<ModeBar {...defaultProps} mode="wellness" />)
    expect(screen.getAllByText('🧘 Wellness').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Sleep Story').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Meditation').length).toBeGreaterThanOrEqual(1)
  })

  it('calls onWellnessTypeChange when a wellness chip is clicked', () => {
    const onWellnessTypeChange = vi.fn()
    render(<ModeBar {...defaultProps} mode="wellness" onWellnessTypeChange={onWellnessTypeChange} />)
    fireEvent.click(screen.getAllByText('Breathing')[0])
    expect(onWellnessTypeChange).toHaveBeenCalledWith('Breathing')
  })

  it('hides wellness chips in chat mode', () => {
    render(<ModeBar {...defaultProps} mode="chat" />)
    expect(screen.queryAllByText('Sleep Story')).toHaveLength(0)
    expect(screen.queryAllByText('Meditation')).toHaveLength(0)
  })

  it('shows create mode button', () => {
    render(<ModeBar {...defaultProps} mode="chat" />)
    expect(screen.getAllByText('🎨 Create').length).toBeGreaterThanOrEqual(1)
  })

  it('shows read mode button', () => {
    render(<ModeBar {...defaultProps} mode="chat" />)
    expect(screen.getAllByText('📄 Read').length).toBeGreaterThanOrEqual(1)
  })

  it('shows talk mode button', () => {
    render(<ModeBar {...defaultProps} mode="chat" />)
    expect(screen.getAllByText('🎙️ Talk').length).toBeGreaterThanOrEqual(1)
  })

  it('shows style chips in create mode', () => {
    render(<ModeBar {...defaultProps} mode="create" />)
    expect(screen.getAllByText('Realistic').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Cartoon').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Watercolor').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Sketch').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Fantasy').length).toBeGreaterThanOrEqual(1)
  })

  it('calls onCreateStyleChange when a style chip is clicked', () => {
    const onCreateStyleChange = vi.fn()
    render(<ModeBar {...defaultProps} mode="create" onCreateStyleChange={onCreateStyleChange} />)
    fireEvent.click(screen.getAllByText('Watercolor')[0])
    expect(onCreateStyleChange).toHaveBeenCalledWith('Watercolor')
  })

  it('highlights selected style', () => {
    render(<ModeBar {...defaultProps} mode="create" createStyle="Fantasy" />)
    expect(screen.getAllByText('Fantasy')[0]).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getAllByText('Realistic')[0]).toHaveAttribute('aria-pressed', 'false')
  })

  it('hides style chips in chat mode', () => {
    render(<ModeBar {...defaultProps} mode="chat" />)
    expect(screen.queryAllByText('Realistic')).toHaveLength(0)
    expect(screen.queryAllByText('Cartoon')).toHaveLength(0)
  })
})
