import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, renderHook } from '@testing-library/react'
import React from 'react'

import { ChatProvider, useChatHealth, useChatModel, useChatUI, useChatContext } from './ChatContext'

const healthValue = {
  health: { status: 'healthy', model_loaded: true, model_type: 'gpt2', summary: 'ok', inference_count: 3, is_inferencing: false } as never,
  refreshHealth: vi.fn(),
}

const modelValue = {
  model: 'gpt2',
  setModel: vi.fn(),
  availableModels: ['gpt2'],
  modelInfoMap: {},
  temperature: 0.8,
  setTemperature: vi.fn(),
  maxTokens: 256,
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
}

const uiValue = {
  onOpenSettings: vi.fn(),
  onOpenShortcuts: vi.fn(),
  onOpenConversationViewer: vi.fn(),
  showToast: vi.fn(),
}

function renderProvider() {
  return render(
    <ChatProvider health={healthValue} model={modelValue} ui={uiValue}>
      <div>child content</div>
    </ChatProvider>
  )
}

afterEach(() => cleanup())

describe('ChatContext', () => {
  it('renders children inside the provider', () => {
    renderProvider()
    expect(screen.getByText('child content')).toBeDefined()
  })

  it('useChatHealth throws outside provider', () => {
    expect(() => renderHook(() => useChatHealth())).toThrow('useChatHealth must be used within ChatProvider')
  })

  it('useChatModel throws outside provider', () => {
    expect(() => renderHook(() => useChatModel())).toThrow('useChatModel must be used within ChatProvider')
  })

  it('useChatUI throws outside provider', () => {
    expect(() => renderHook(() => useChatUI())).toThrow('useChatUI must be used within ChatProvider')
  })

  it('useChatContext throws outside provider', () => {
    expect(() => renderHook(() => useChatContext())).toThrow()
  })

  it('useChatHealth exposes the health value', () => {
    const { result } = renderHook(() => useChatHealth(), { wrapper: ({ children }: any) => (
      <ChatProvider health={healthValue} model={modelValue} ui={uiValue}>{children}</ChatProvider>
    ) })
    const h = result.current.health
    if (h && typeof h === 'object') {
      expect(h.model_type).toBe('gpt2')
      expect(h.model_loaded).toBe(true)
    }
    result.current.refreshHealth()
    expect(healthValue.refreshHealth).toHaveBeenCalled()
  })

  it('useChatModel exposes the model value', () => {
    const { result } = renderHook(() => useChatModel(), { wrapper: ({ children }: any) => (
      <ChatProvider health={healthValue} model={modelValue} ui={uiValue}>{children}</ChatProvider>
    ) })
    expect(result.current.model).toBe('gpt2')
    expect(result.current.temperature).toBe(0.8)
    expect(result.current.maxTokens).toBe(256)
    result.current.setModel('qwen')
    expect(modelValue.setModel).toHaveBeenCalledWith('qwen')
    result.current.handleSelectSoul({} as never)
    expect(modelValue.handleSelectSoul).toHaveBeenCalled()
    result.current.setInput((prev: string) => prev + '!')
    expect(modelValue.setInput).toHaveBeenCalled()
  })

  it('useChatUI exposes UI callbacks', () => {
    const { result } = renderHook(() => useChatUI(), { wrapper: ({ children }: any) => (
      <ChatProvider health={healthValue} model={modelValue} ui={uiValue}>{children}</ChatProvider>
    ) })
    result.current.onOpenSettings()
    result.current.onOpenShortcuts()
    result.current.onOpenConversationViewer()
    result.current.showToast('hi', 'success')
    expect(uiValue.onOpenSettings).toHaveBeenCalled()
    expect(uiValue.onOpenShortcuts).toHaveBeenCalled()
    expect(uiValue.onOpenConversationViewer).toHaveBeenCalled()
    expect(uiValue.showToast).toHaveBeenCalledWith('hi', 'success')
  })

  it('useChatContext merges all three sub-contexts', () => {
    const { result } = renderHook(() => useChatContext(), { wrapper: ({ children }: any) => (
      <ChatProvider health={healthValue} model={modelValue} ui={uiValue}>{children}</ChatProvider>
    ) })
    expect(result.current.model).toBe('gpt2')
    const h = result.current.health
    if (h && typeof h === 'object') {
      expect(h.model_loaded).toBe(true)
    }
    expect(typeof result.current.showToast).toBe('function')
    expect(typeof result.current.refreshHealth).toBe('function')
    expect(typeof result.current.handleSelectModel).toBe('function')
  })

  it('provider allows nested independent consumption', () => {
    function Consumer() {
      const health = useChatHealth()
      const ui = useChatUI()
      return <button onClick={() => { health.refreshHealth(); ui.showToast('x') }}>combined</button>
    }
    render(
      <ChatProvider health={healthValue} model={modelValue} ui={uiValue}>
        <Consumer />
      </ChatProvider>
    )
    fireEvent.click(screen.getByText('combined'))
    expect(healthValue.refreshHealth).toHaveBeenCalled()
    expect(uiValue.showToast).toHaveBeenCalledWith('x')
  })
})
