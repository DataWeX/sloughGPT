import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: vi.fn((...args: any[]) => args.join(' ')),
  IconX: () => <span data-testid="icon-x">x</span>,
  IconEye: () => <span data-testid="icon-eye">eye</span>,
  IconStar: () => <span data-testid="icon-star">star</span>,
  IconTrash: () => <span data-testid="icon-trash">trash</span>,
  Button: ({ children, onClick, variant, size, className, ...rest }: any) => (
    <button onClick={onClick} className={className} data-variant={variant} data-size={size} {...rest}>{children}</button>
  ),
}))

vi.mock('./VisionTabContent', () => ({
  VisionTabContent: (props: any) => (
    <div data-testid="vision-tab-content">
      Vision Tab
      <button onClick={() => props.onGeneratedImage?.('data:img', 'test')}>Generate</button>
    </div>
  ),
}))

vi.mock('./KnowledgeTab', () => ({
  KnowledgeTab: (props: any) => (
    <div data-testid="knowledge-tab">
      <button onClick={props.onOpenConversationViewer}>View Log</button>
      <button onClick={props.onOpenSettings}>Settings</button>
    </div>
  ),
}))

vi.mock('./QuickPrompts', () => ({
  QuickPrompts: ({ onUsePrompt }: { onUsePrompt: (text: string) => void }) => (
    <div data-testid="quick-prompts">
      <button onClick={() => onUsePrompt('test prompt')}>Test Prompt</button>
    </div>
  ),
}))

vi.mock('./ChatBookmarksPanel', () => ({
  ChatBookmarksPanel: ({ bookmarks }: { bookmarks: any[] }) => (
    <div data-testid="chat-bookmarks-panel">Bookmarks: {bookmarks.length}</div>
  ),
}))

const mockUseChatContext = vi.fn()
vi.mock('@/contexts/ChatContext', () => ({
  useChatContext: () => mockUseChatContext(),
}))

import { ChatToolPanel } from './ChatToolPanel'

const defaultCtx = {
  health: { status: 'healthy', model_loaded: true, model_type: 'gpt2', uptime_seconds: 100,
    request_count: 10, error_count: 0, inference_count: 5, total_tokens: 1000,
    tokens_per_sec: 10, avg_tokens_per_request: 200, avg_latency_ms: 100, requests_per_minute: 5 },
  refreshHealth: vi.fn(),
  model: 'gpt2',
  setModel: vi.fn(),
  availableModels: ['gpt2'],
  modelInfoMap: {},
  temperature: 0.7,
  setTemperature: vi.fn(),
  maxTokens: 200,
  setMaxTokens: vi.fn(),
  loadingModel: null,
  handleSelectModel: vi.fn(),
  handleUnloadModel: vi.fn(),
  souls: [],
  currentSoul: null,
  setCurrentSoul: vi.fn(),
  handleSelectSoul: vi.fn(),
  checkpoints: [],
  currentCheckpoint: undefined,
  setCurrentCheckpoint: vi.fn(),
  onLoadCheckpoint: vi.fn(),
  agents: [],
  currentAgent: null,
  setCurrentAgent: vi.fn(),
  visionCaps: null,
  visionCaptionHistory: [],
  visionVocabSize: undefined,
  learnerInfo: null,
  learnerTraining: false,
  setLearnerInfo: vi.fn(),
  setLearnerTraining: vi.fn(),
  onTrainStep: vi.fn(),
  setInput: vi.fn(),
  onOpenSettings: vi.fn(),
  onOpenShortcuts: vi.fn(),
  onOpenConversationViewer: vi.fn(),
  showToast: vi.fn(),
}

describe('ChatToolPanel', () => {
  const onClose = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
    mockUseChatContext.mockReturnValue(defaultCtx)
  })
  afterEach(cleanup)

  it('renders header with Tools label when open', () => {
    render(<ChatToolPanel open={true} onClose={onClose} sessionId="s1" />)
    expect(screen.getByText('Tools')).toBeDefined()
    expect(screen.getByTestId('knowledge-tab')).toBeDefined()
    expect(screen.getByTestId('quick-prompts')).toBeDefined()
    expect(screen.getByTestId('chat-bookmarks-panel')).toBeDefined()
  })

  it('renders nothing when closed', () => {
    const { container } = render(<ChatToolPanel open={false} onClose={onClose} sessionId="s1" />)
    const panel = container.querySelector('#chat-tool-panel')
    expect(panel?.className).toContain('w-0')
    expect(screen.queryByText('Tools')).toBeNull()
  })

  it('switches to vision mode when eye icon clicked', () => {
    render(<ChatToolPanel open={true} onClose={onClose} sessionId="s1" />)
    fireEvent.click(screen.getByLabelText('Show vision panel'))
    expect(screen.getByText('Vision')).toBeDefined()
    expect(screen.getByTestId('vision-tab-content')).toBeDefined()
  })

  it('switches back to tools mode', () => {
    render(<ChatToolPanel open={true} onClose={onClose} sessionId="s1" />)
    fireEvent.click(screen.getByLabelText('Show vision panel'))
    expect(screen.getByText('Vision')).toBeDefined()
    fireEvent.click(screen.getByLabelText('Show tools panel'))
    expect(screen.getByText('Tools')).toBeDefined()
  })

  it('calls onClose when close button clicked', () => {
    render(<ChatToolPanel open={true} onClose={onClose} sessionId="s1" />)
    fireEvent.click(screen.getByLabelText('Close tools panel'))
    expect(onClose).toHaveBeenCalledTimes(1)
  })

  it('calls onOpenConversationViewer from footer', () => {
    render(<ChatToolPanel open={true} onClose={onClose} sessionId="s1" />)
    fireEvent.click(screen.getByText('Log'))
    expect(defaultCtx.onOpenConversationViewer).toHaveBeenCalled()
  })

  it('calls onOpenSettings from footer', () => {
    render(<ChatToolPanel open={true} onClose={onClose} sessionId="s1" />)
    const settingsBtns = screen.getAllByText('Settings')
    const footerBtn = settingsBtns.find(b => b.closest('[id="chat-tool-panel"]')?.querySelector('svg'))
    fireEvent.click(footerBtn!)
    expect(defaultCtx.onOpenSettings).toHaveBeenCalled()
  })

  it('calls onOpenShortcuts from footer', () => {
    render(<ChatToolPanel open={true} onClose={onClose} sessionId="s1" />)
    fireEvent.click(screen.getByText('Keys'))
    expect(defaultCtx.onOpenShortcuts).toHaveBeenCalled()
  })

  it('handles QuickPrompts onUsePrompt', () => {
    render(<ChatToolPanel open={true} onClose={onClose} sessionId="s1" />)
    fireEvent.click(screen.getByText('Test Prompt'))
    expect(defaultCtx.setInput).toHaveBeenCalledWith('test prompt')
  })
})
