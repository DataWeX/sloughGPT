// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: vi.fn((...args: any[]) => args.join(' ')),
  IconRefresh: () => <span data-testid="icon-refresh">refresh</span>,
}))

const hoisted = vi.hoisted(() => ({
  chatController: { inspectContext: vi.fn() },
  soulsController: { getModes: vi.fn(), getTraitWeights: vi.fn() },
  feedbackController: { getFeedbackStats: vi.fn() },
  chatDB: { getKnowledge: vi.fn() },
  logger: { debug: vi.fn() },
}))

vi.mock('@/lib/chat-controller', () => ({
  chatController: hoisted.chatController,
}))

vi.mock('@/lib/souls-controller', () => ({
  soulsController: hoisted.soulsController,
}))

vi.mock('@/lib/feedback-controller', () => ({
  feedbackController: hoisted.feedbackController,
}))

vi.mock('@/lib/db', () => ({
  chatDB: hoisted.chatDB,
}))

vi.mock('@/lib/dev-log', () => ({
  logger: hoisted.logger,
}))

import { ContextTab } from './ContextTab'

const inspector = {
  system_prompt: 'You are a warm and curious companion.',
  session_messages: [{ role: 'user', content: 'hi' }],
  working_memory: [{ k: 'v' }],
  semantic_keys: ['user prefers espresso'],
  episodic_count: 3,
  sensory_buffer_size: 2,
  frame_history_size: 4,
  last_frame: null,
}

const modes = {
  personality: { label: 'Warm', confidence: 0.8 },
  memory: { label: 'Expansive', confidence: 0.6, capacity: 8 },
  style: { label: 'Casual', confidence: 0.5 },
  task: { label: 'Balanced', confidence: 0.7 },
}

const traits = {
  personality: { warmth: 0.9, curiosity: 0.7, confidence: 0.6 },
  cognition: { planning: 0.5 },
  emotion: {},
}

const feedbackStats = {
  db_stats: { conversations: 2, messages: 10, feedback_total: 5, thumbs_up: 4, thumbs_down: 1, ratio: 0.8 },
  current_weights: { temperature: 0.7, repetition_penalty: 1.1 },
  history_length: 5,
}

describe('ContextTab', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    hoisted.chatController.inspectContext.mockResolvedValue(inspector)
    hoisted.soulsController.getModes.mockResolvedValue(modes)
    hoisted.soulsController.getTraitWeights.mockResolvedValue(traits)
    hoisted.feedbackController.getFeedbackStats.mockResolvedValue(feedbackStats)
    hoisted.chatDB.getKnowledge.mockResolvedValue([{ id: 'k1', content: 'fact', timestamp: 1 }])
  })
  afterEach(cleanup)

  it('renders steering modes with labels and confidence bars', async () => {
    render(<ContextTab />)
    expect((await screen.findAllByText('Personality')).length).toBeGreaterThan(0)
    expect(screen.getByText('Warm')).toBeDefined()
    expect(screen.getByText('Memory')).toBeDefined()
    expect(screen.getByText('Expansive')).toBeDefined()
    expect(screen.getByText('Style')).toBeDefined()
    expect(screen.getByText('Task')).toBeDefined()
  })

  it('renders trait weights with percentages', async () => {
    render(<ContextTab />)
    await screen.findByText('Trait weights')
    expect(screen.getByText('warmth')).toBeDefined()
    expect(screen.getByText('90%')).toBeDefined()
    expect(screen.getByText('curiosity')).toBeDefined()
    expect(screen.getByText('70%')).toBeDefined()
  })

  it('renders workspace memory counts from the inspector', async () => {
    render(<ContextTab />)
    await screen.findByText('Workspace memory')
    expect(screen.getByText('Working')).toBeDefined()
    expect(screen.getByText('Semantic')).toBeDefined()
    expect(screen.getByText('Episodic')).toBeDefined()
    expect(screen.getByText('Sensory')).toBeDefined()
    expect(screen.getByText('Context frames: 4')).toBeDefined()
  })

  it('shows and hides the system prompt on toggle', async () => {
    render(<ContextTab />)
    await screen.findByText('Show system prompt')
    expect(screen.queryByText(/warm and curious companion/)).toBeNull()
    fireEvent.click(screen.getByText('Show system prompt'))
    expect(screen.getByText(/warm and curious companion/)).toBeDefined()
    fireEvent.click(screen.getByText('Hide system prompt'))
    expect(screen.queryByText(/warm and curious companion/)).toBeNull()
  })

  it('refreshes all sources on refresh click', async () => {
    render(<ContextTab />)
    await screen.findAllByText('Personality')
    fireEvent.click(screen.getByLabelText('Refresh context'))
    await waitFor(() => expect(hoisted.chatController.inspectContext).toHaveBeenCalledTimes(2))
    await waitFor(() => expect(hoisted.soulsController.getModes).toHaveBeenCalledTimes(2))
    await waitFor(() => expect(hoisted.soulsController.getTraitWeights).toHaveBeenCalledTimes(2))
    await waitFor(() => expect(hoisted.chatDB.getKnowledge).toHaveBeenCalledTimes(2))
    await waitFor(() => expect(hoisted.feedbackController.getFeedbackStats).toHaveBeenCalledTimes(2))
  })

  it('renders injected knowledge and feedback stats in Inputs', async () => {
    render(<ContextTab />)
    await screen.findByText('Inputs')
    expect(screen.getByText('Knowledge')).toBeDefined()
    expect(screen.getByText('1 snippet')).toBeDefined()
    expect(screen.getByText('Feedback')).toBeDefined()
    expect(screen.getByText('4 up · 1 down')).toBeDefined()
  })

  it('pluralizes the knowledge snippet count', async () => {
    hoisted.chatDB.getKnowledge.mockResolvedValue([{ id: 'k1', content: 'a', timestamp: 1 }, { id: 'k2', content: 'b', timestamp: 2 }])
    render(<ContextTab />)
    await screen.findByText('Inputs')
    expect(screen.getByText('2 snippets')).toBeDefined()
  })

  it('still renders when knowledge and feedback fail', async () => {
    hoisted.chatDB.getKnowledge.mockRejectedValue(new Error('db down'))
    hoisted.feedbackController.getFeedbackStats.mockRejectedValue(new Error('api down'))
    render(<ContextTab />)
    expect(await screen.findByText('Workspace memory')).toBeDefined()
    expect(screen.queryByText('Inputs')).toBeNull()
  })

  it('shows the unavailable state when everything fails', async () => {
    hoisted.chatController.inspectContext.mockResolvedValue(null)
    hoisted.soulsController.getModes.mockRejectedValue(new Error('down'))
    hoisted.soulsController.getTraitWeights.mockRejectedValue(new Error('down'))
    hoisted.chatDB.getKnowledge.mockRejectedValue(new Error('down'))
    hoisted.feedbackController.getFeedbackStats.mockRejectedValue(new Error('down'))
    render(<ContextTab />)
    expect(await screen.findByText(/Context unavailable right now/)).toBeDefined()
  })

  it('keeps showing available sources when only one fails', async () => {
    hoisted.soulsController.getTraitWeights.mockRejectedValue(new Error('down'))
    render(<ContextTab />)
    await screen.findAllByText('Personality')
    expect(screen.getByText('Workspace memory')).toBeDefined()
    expect(screen.queryByText('Trait weights')).toBeNull()
  })
})
