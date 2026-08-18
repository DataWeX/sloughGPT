import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, waitFor, fireEvent } from '@testing-library/react'
import React from 'react'
import { act } from 'react'

const mockCva = vi.hoisted(() => { const fn = () => ''; return fn })
vi.mock('class-variance-authority', () => ({ cva: () => mockCva }))

vi.mock('@sloughgpt/strui', () => {
  const iconMock = (name: string) => { const C = () => <span data-testid={`icon-${name}`}>{name}</span>; C.displayName = `Icon${name}`; return C }
  const passthrough = ({ children, className }: any) => <div className={className}>{children}</div>
  return {
    cn: vi.fn((...args: any[]) => args.join(' ')),
    Card: passthrough,
    CardContent: passthrough,
    CardHeader: passthrough,
    CardDescription: ({ children, className }: any) => <p className={className}>{children}</p>,
    CardTitle: ({ children, className }: any) => <div className={className}>{children}</div>,
    Button: ({ children, onClick, type, disabled, className, 'aria-label': ariaLabel }: any) => (
      <button onClick={onClick} type={type} disabled={disabled} className={className} aria-label={ariaLabel}>{children}</button>
    ),
    IconChevronRight: iconMock('chevron-right'),
    IconMessage: iconMock('message'),
    IconSearch: iconMock('search'),
    IconBolt: iconMock('bolt'),
    IconChart: iconMock('chart'),
    LossCurve: (props: Record<string, unknown>) => null,
  }
})

vi.mock('@/components/icons/NavIcons', () => ({
  IconChat: () => <span data-testid="icon-chat">chat</span>,
  IconModels: () => <span data-testid="icon-models">models</span>,
    Skeleton: ({ className }: any) => <div className={className} data-testid="skeleton" />,
}))

vi.mock('next/link', () => ({
  default: ({ children, href, className, ...rest }: any) => <a href={href} className={className} {...rest}>{children}</a>,
}))

const { mockPush, mockApiGet, mockChatSend, mockKnowledgeAdd, mockSessionList, mockDatasetList, mockAddToast } = vi.hoisted(() => ({
  mockPush: vi.fn(),
  mockApiGet: vi.fn(),
  mockChatSend: vi.fn(),
  mockKnowledgeAdd: vi.fn(),
  mockSessionList: vi.fn(),
  mockDatasetList: vi.fn(),
  mockAddToast: vi.fn(),
}))

const state = vi.hoisted(() => ({ health: null as any, liveHealth: null as any, home: null as any }))

const homeTranslations: Record<string, string> = {
  'home.subtitle.offline': 'Server offline',
  'home.apiOffline.title': 'API not reachable',
  'home.apiOffline.body': 'API server at {url} is not reachable',
  'home.stats.status': 'Status',
  'home.stats.models': 'Models',
  'home.stats.personality': 'Personality',
}

const mockT = vi.fn((key: string, params?: Record<string, string | number>) => {
  let text = homeTranslations[key] ?? key
  if (params) {
    for (const [k, v] of Object.entries(params)) text = text.replace(`{${k}}`, String(v))
  }
  return text
})

vi.mock('next/navigation', () => ({ useRouter: () => ({ push: mockPush }) }))
vi.mock('@/hooks/useLocale', () => ({ useLocale: () => ({ t: mockT, locale: 'en', setLocale: vi.fn(), locales: ['en'] }) }))
vi.mock('@/hooks/useLiveStatus', () => ({ useLiveStatus: () => ({ healthLegacy: state.health, health: state.liveHealth }) }))
vi.mock('@/hooks/useHomePageData', async () => {
  const React = await import('react')
  return {
    useHomePageData: () => {
      const [testRunning, setTestRunning] = React.useState(false)
      const [testResponse, setTestResponse] = React.useState<string | null>(null)
      const [knowledgeCount, setKnowledgeCount] = React.useState(state.home?.knowledgeCount ?? 0)
      return { ...state.home, testRunning, setTestRunning, testResponse, setTestResponse, knowledgeCount, setKnowledgeCount }
    },
  }
})
vi.mock('@/lib/toast-store', () => ({ useToastStore: (sel: any) => sel({ addToast: mockAddToast }) }))
vi.mock('@/lib/http-client', () => ({ apiGet: mockApiGet }))
vi.mock('@/lib/chat-controller', () => ({ chatController: { send: mockChatSend } }))
vi.mock('@/lib/knowledge-controller', () => ({ knowledgeController: { add: mockKnowledgeAdd } }))
vi.mock('@/lib/session-controller', () => ({ sessionController: { list: mockSessionList } }))
vi.mock('@/lib/dataset-controller', () => ({ datasetController: { list: mockDatasetList } }))

import HomePage from './page'

const onlineHealth = { model_loaded: true, model_type: 'hf/gpt2', inference_count: 5 }

function makeHomeData(overrides: Record<string, unknown> = {}) {
  return {
    modelCount: 3,
    checkpointCount: 0,
    modelStatus: { loaded: true, model: 'gpt2' },
    currentSoul: { name: 'Warm', description: '', traits: [] },
    recentSessions: [],
    runningTraining: null,
    knowledgeCount: 0,
    recentJobs: [],
    testRunning: false,
    testResponse: null,
    setTestRunning: vi.fn(),
    setTestResponse: vi.fn(),
    setKnowledgeCount: vi.fn(),
    inferenceCount: 5,
    healthSummary: 'hf/gpt2',
    feedbackStats: null,
    errors: { models: false, soul: false, sessions: false, training: false, knowledge: false, feedback: false },
    ...overrides,
  }
}

afterEach(() => { cleanup(); vi.useRealTimers() })
beforeEach(() => {
  vi.clearAllMocks()
  localStorage.clear()
  state.health = onlineHealth
  state.liveHealth = onlineHealth
  state.home = makeHomeData()
  mockSessionList.mockResolvedValue([])
  mockDatasetList.mockResolvedValue([])
  mockApiGet.mockResolvedValue(null)
})

describe('HomePage', () => {
  it('shows connecting subtitle and skeletons while health is loading', () => {
    state.health = null
    state.liveHealth = null
    render(<HomePage />)
    expect(screen.getByText('Connecting...')).toBeTruthy()
    expect(document.querySelector('.animate-pulse')).toBeTruthy()
    expect(screen.queryByText('Online')).toBeFalsy()
  })

  it('shows the API offline card with the configured URL', () => {
    state.health = 'offline'
    state.liveHealth = 'offline'
    render(<HomePage />)
    expect(screen.getByText('API server at http://localhost:8000 is not reachable')).toBeTruthy()
    expect(screen.queryByText('Online')).toBeFalsy()
  })

  it('shows startup progress card while offline', async () => {
    state.health = 'offline'
    state.liveHealth = 'offline'
    mockApiGet.mockResolvedValue({ phase: 'loading-model', step: 2, total: 5, message: 'Loading PyTorch' })
    render(<HomePage />)
    await waitFor(() => { expect(mockApiGet).toHaveBeenCalledWith('/health/startup-progress') })
    await waitFor(() => { expect(screen.getByText('Starting up… (2/5)')).toBeTruthy() })
    expect(screen.getByText('Loading PyTorch')).toBeTruthy()
  })

  it('renders a time-aware greeting', async () => {
    vi.useFakeTimers({ toFake: ['Date'] })
    vi.setSystemTime(new Date(2026, 5, 15, 9, 30, 0))
    render(<HomePage />)
    await waitFor(() => { expect(screen.getByText('Good morning')).toBeTruthy() })
  })

  it('summarizes the loaded model and conversation count in the subtitle', () => {
    state.home = makeHomeData({ healthSummary: 'hf/gpt2', inferenceCount: 5 })
    render(<HomePage />)
    expect(screen.getByText('gpt2 loaded · 5 conversations')).toBeTruthy()
  })

  it('renders stat cards with model count, soul, and active model', () => {
    state.home = makeHomeData({
      modelCount: 3,
      currentSoul: { name: 'Warm', description: '', traits: [] },
      modelStatus: { loaded: true, model: 'gpt2' },
      inferenceCount: 5,
    })
    render(<HomePage />)
    expect(screen.getAllByText('3').length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText('Warm')).toBeTruthy()
    expect(screen.getByText('gpt2 + Warm')).toBeTruthy()
    expect(screen.getAllByText('5 conversations').length).toBeGreaterThanOrEqual(1)
  })

  it('shows Not loaded and hides quick actions when the model is unloaded', () => {
    state.home = makeHomeData({ modelStatus: { loaded: false, model: null } })
    render(<HomePage />)
    expect(screen.getByText('Not loaded')).toBeTruthy()
    expect(screen.queryByText('Test model')).toBeFalsy()
    expect(screen.queryByText('Quick note')).toBeFalsy()
  })

  it('runs a quick test against chatController and shows the response', async () => {
    let resolveTest: (v: { message: string }) => void
    mockChatSend.mockReturnValue(new Promise(res => { resolveTest = res }))
    render(<HomePage />)
    await act(async () => { screen.getByText('Test model').click() })
    expect(mockChatSend).toHaveBeenCalledWith('Hello!')
    expect(screen.getByText('Testing...')).toBeTruthy()
    await act(async () => { resolveTest!({ message: 'Hello from the model' }) })
    await waitFor(() => { expect(screen.getByText('Hello from the model')).toBeTruthy() })
  })

  it('shows an extracted error message when the quick test fails', async () => {
    mockChatSend.mockRejectedValue(new Error('connection refused'))
    render(<HomePage />)
    await act(async () => { screen.getByText('Test model').click() })
    await waitFor(() => { expect(screen.getByText('connection refused')).toBeTruthy() })
  })

  it('saves a quick note via knowledgeController and toasts success', async () => {
    mockKnowledgeAdd.mockResolvedValue({})
    render(<HomePage />)
    const input = screen.getByPlaceholderText('e.g., I prefer Python over JavaScript')
    await act(async () => { fireEvent.change(input, { target: { value: 'I like coffee' } }) })
    await act(async () => { fireEvent.submit(screen.getByText('Save').closest('form')!) })
    await waitFor(() => { expect(mockKnowledgeAdd).toHaveBeenCalledWith('I like coffee', 'general') })
    expect(mockAddToast).toHaveBeenCalledWith('Fact saved', 'success')
  })

  it('toasts failure when saving a quick note errors', async () => {
    mockKnowledgeAdd.mockRejectedValue(new Error('boom'))
    render(<HomePage />)
    const input = screen.getByPlaceholderText('e.g., I prefer Python over JavaScript')
    await act(async () => { fireEvent.change(input, { target: { value: 'I like coffee' } }) })
    await act(async () => { fireEvent.submit(screen.getByText('Save').closest('form')!) })
    await waitFor(() => { expect(mockAddToast).toHaveBeenCalledWith('Failed to save', 'error') })
  })

  it('renders feedback stats and links to training', () => {
    state.home = makeHomeData({
      feedbackStats: { db_stats: { feedback_total: 10, thumbs_up: 7, thumbs_down: 3, ratio: 0.7 }, train_stats: null, adapter_stats: null },
    })
    render(<HomePage />)
    expect(screen.getByText('10')).toBeTruthy()
    expect(screen.getByText('👍 7')).toBeTruthy()
    expect(screen.getByText('👎 3')).toBeTruthy()
    expect(screen.getByText('70% positive')).toBeTruthy()
    const link = screen.getByText('Train from feedback →')
    expect(link.getAttribute('href')).toBe('/training')
  })

  it('renders the running training card', () => {
    state.home = makeHomeData({ runningTraining: { name: 'lora-finetune', status_message: 'Epoch 2/5' } })
    render(<HomePage />)
    expect(screen.getByText('Training: lora-finetune')).toBeTruthy()
    expect(screen.getByText('Epoch 2/5')).toBeTruthy()
  })

  it('lists recent sessions and jobs and navigates on click', async () => {
    state.home = makeHomeData({
      recentSessions: [
        { id: 's1', name: 'First chat', updated_at: new Date().toISOString(), message_count: 3 },
        { id: 's2', name: 'Second chat', updated_at: new Date(Date.now() - 60 * 60000).toISOString() },
      ],
      recentJobs: [{ id: 'j1', name: 'distill', status: 'completed' }, { id: 'j2', name: 'fine-tune', status: 'running' }],
    })
    render(<HomePage />)
    expect(screen.getAllByText('First chat').length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText('Second chat')).toBeTruthy()
    expect(screen.getByText('distill')).toBeTruthy()
    expect(screen.getByText('fine-tune')).toBeTruthy()
    await act(async () => { screen.getAllByText('First chat')[0].click() })
    expect(mockPush).toHaveBeenCalledWith('/chat?session=s1')
  })

  it('provides a resume button for the most recent session', async () => {
    state.home = makeHomeData({
      recentSessions: [{ id: 's9', name: 'Latest chat', updated_at: new Date().toISOString(), message_count: 4 }],
    })
    render(<HomePage />)
    const resume = screen.getAllByText('Latest chat')[0]
    await act(async () => { resume.click() })
    expect(mockPush).toHaveBeenCalledWith('/chat?session=s9')
  })

  it('dismisses the onboarding card via localStorage', async () => {
    render(<HomePage />)
    expect(screen.getByText('New here?')).toBeTruthy()
    await act(async () => {})
    const gotIt = screen.getByText('Got it')
    await act(async () => { gotIt.click() })
    expect(localStorage.getItem('onboarding_dismissed')).toBe('1')
    expect(screen.queryByText('New here?')).toBeFalsy()
  })

  it('renders conversation and dataset stats cards from controller data', async () => {
    mockSessionList.mockResolvedValue([
      { id: 'a', name: 'A', updated_at: '2026-06-01T10:00:00Z', messages: [
        { content: 'hello world foo', timestamp: '2026-06-01T10:00:00Z' },
        { content: 'hi', timestamp: '2026-06-01T10:00:00Z' },
      ] },
    ])
    mockDatasetList.mockResolvedValue([
      { id: 'd1', size: 1048576, samples: 100 },
    ])
    render(<HomePage />)
    await waitFor(() => { expect(screen.getByText('Your stats')).toBeTruthy() })
    expect(screen.getAllByText('1').length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText('Words')).toBeTruthy()
    expect(screen.getAllByText('Datasets').length).toBeGreaterThanOrEqual(2)
    expect(screen.getByText('1.0 MB')).toBeTruthy()
  })

  it('renders the live health system card with CPU, memory, requests, and uptime', () => {
    state.liveHealth = { cpu_percent: 42.3, memory_percent: 60.9, request_count: 1200, uptime_seconds: 3661 }
    render(<HomePage />)
    expect(screen.getByText('42%')).toBeTruthy()
    expect(screen.getByText('61%')).toBeTruthy()
    expect(screen.getByText('1,200')).toBeTruthy()
    expect(screen.getByText('1h 1m')).toBeTruthy()
  })

  it('always shows the CTA grid links', () => {
    render(<HomePage />)
    expect(screen.getByText('Start chatting').closest('a')?.getAttribute('href')).toBe('/chat')
    expect(screen.getByText('Personalities').closest('a')?.getAttribute('href')).toBe('/models')
    expect(screen.getByText('Datasets').closest('a')?.getAttribute('href')).toBe('/datasets')
    expect(screen.getByText('Teach me').closest('a')?.getAttribute('href')).toBe('/training')
    expect(screen.getByText('System Health').closest('a')?.getAttribute('href')).toBe('/monitoring')
    expect(screen.getByText('Knowledge').closest('a')?.getAttribute('href')).toBe('/knowledge')
  })
})
