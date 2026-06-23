// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import React from 'react'
import { ChatProvider } from '@/contexts/ChatContext'

vi.mock('./KnowledgeTab', () => ({ KnowledgeTab: () => <div data-testid="knowledge-tab" /> }))
vi.mock('./VisionTabContent', () => ({ VisionTabContent: () => <div data-testid="vision-tab" /> }))
vi.mock('./QuickPrompts', () => ({ QuickPrompts: () => <div data-testid="quick-prompts" /> }))

import { ChatToolPanel } from './ChatToolPanel'

const healthCtx = { health: { model_loaded: true, model_type: 'gpt2' } as any, refreshHealth: vi.fn() }
const modelCtx = {
  model: 'gpt2', setModel: vi.fn(), availableModels: [], modelInfoMap: {},
  temperature: 0.8, setTemperature: vi.fn(), maxTokens: 200, setMaxTokens: vi.fn(),
  loadingModel: null, handleSelectModel: vi.fn(), handleUnloadModel: vi.fn(),
}
const uiCtx = { onOpenSettings: vi.fn(), onOpenShortcuts: vi.fn(), onOpenConversationViewer: vi.fn(), showToast: vi.fn() }

function wrap(ui: React.ReactElement) {
  return <ChatProvider health={healthCtx} model={modelCtx} ui={uiCtx}>{ui}</ChatProvider>
}

describe('ChatToolPanel', () => {
  afterEach(cleanup)

  it('renders tools header when open', () => {
    render(wrap(<ChatToolPanel open={true} onClose={vi.fn()} sessionId="s1" />))
    expect(screen.getByText('Tools')).toBeDefined()
  })

  it('does not render when closed', () => {
    const { container } = render(wrap(<ChatToolPanel open={false} onClose={vi.fn()} sessionId="s1" />))
    expect(container.querySelector('#chat-tool-panel')?.className).toContain('w-0')
  })

  it('renders knowledge tab by default', () => {
    render(wrap(<ChatToolPanel open={true} onClose={vi.fn()} sessionId="s1" />))
    expect(screen.getByTestId('knowledge-tab')).toBeDefined()
  })

  it('toggles to vision tab when eye button clicked', () => {
    render(wrap(<ChatToolPanel open={true} onClose={vi.fn()} sessionId="s1" />))
    fireEvent.click(screen.getByLabelText('Show vision panel'))
    expect(screen.getByText('Vision')).toBeDefined()
    expect(screen.getByTestId('vision-tab')).toBeDefined()
  })

  it('calls onClose when close button clicked', () => {
    const onClose = vi.fn()
    render(wrap(<ChatToolPanel open={true} onClose={onClose} sessionId="s1" />))
    fireEvent.click(screen.getByLabelText('Close tools panel'))
    expect(onClose).toHaveBeenCalled()
  })
})
