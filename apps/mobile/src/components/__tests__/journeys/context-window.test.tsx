/**
 * Journey: Mobile: See What the AI Sees (UX_FLOWS.md)
 *
 * In-code journey: "+ Context" note flows through the real chat-store into
 * the /chat/control contract, and the Details sheet reads /context/inspect
 * through the real mapper into the ContextWindowViewer. Only the transport
 * is mocked. Mapped in domain/journeys/flows.py as m2-context.
 */
import React from 'react'
import AsyncStorage from '@react-native-async-storage/async-storage'
import { fireEvent, render, waitFor } from '../../../test-utils'
import { ContextInjectionBar } from '../../ContextInjectionBar'
import { ChatBottomSheets } from '../../ChatBottomSheets'
import { api } from '../../../services/api-client'
import { useChatStore } from '../../../stores/chat-store'

jest.mock('../../../services/api-client')
jest.mock('../../../services/sse-client')
jest.mock('../../../services/offline-cache')
jest.mock('../../../services/haptics')
jest.mock('../../../services/sounds')
jest.mock('../../../services/toast')
jest.mock('../../../stores/settings-store', () => {
  const state = {
    theme: 'dark',
    temperature: 0.8,
    maxTokens: 256,
    topP: 0.9,
    topK: 40,
    repetitionPenalty: 1.1,
  }
  const useSettingsStore: any = (selector: (s: typeof state) => unknown) => selector(state)
  useSettingsStore.getState = () => state
  useSettingsStore.setState = () => state
  useSettingsStore.subscribe = () => () => {}
  return { useSettingsStore, __esModule: true }
})
jest.mock('../../../stores/hybrid-inference-store', () => ({
  useHybridStore: {
    getState: () => ({
      decideRoute: () => ({ target: 'remote' }),
      executeLocal: jest.fn(),
      offlineOnly: false,
    }),
  },
}))
jest.mock('../../SearchSessionsModal', () => {
  const { Text } = require('react-native')
  return {
    SearchSessionsModal: ({ visible }: any) => (visible ? <Text>search modal</Text> : null),
  }
})

const mockApi = api as jest.Mocked<typeof api>

const INSPECTOR_FIXTURE = {
  system_prompt: 'You are a friendly assistant.',
  session_messages: [
    { role: 'user', content: 'hello there' },
    { role: 'assistant', content: 'Hi! How can I help?' },
  ],
  semantic_keys: ['user_name=Ana'],
}

beforeEach(() => {
  jest.clearAllMocks()
  const store = new Map<string, string>()
  jest.spyOn(AsyncStorage, 'getItem').mockImplementation(async (key) => store.get(key) ?? null)
  jest.spyOn(AsyncStorage, 'setItem').mockImplementation(async (key, value) => {
    store.set(key, String(value))
  })
  ;(mockApi.post as jest.Mock).mockResolvedValue({})
  ;(mockApi.inspectContext as jest.Mock).mockResolvedValue(INSPECTOR_FIXTURE)
  useChatStore.setState({ activeSessionId: 's1', messages: [], sessions: [] })
})

function BarHarness() {
  const injectContext = useChatStore((s: any) => s.injectContext)
  return <ContextInjectionBar onInject={injectContext} />
}

const sheetProps = {
  showInfo: true,
  setShowInfo: jest.fn(),
  showSearchSessions: false,
  setShowSearchSessions: jest.fn(),
  handleSelectSearchSession: jest.fn(),
  showSoulPicker: false,
  setShowSoulPicker: jest.fn(),
  showSettings: false,
  setShowSettings: jest.fn(),
  showChatSettings: false,
  setShowChatSettings: jest.fn(),
  showSystemPrompt: false,
  setShowSystemPrompt: jest.fn(),
  forwardTo: null as any,
  setForwardTo: jest.fn(),
  safeSessions: [{ id: 's1', name: 'Chat 1', created_at: '', updated_at: '', message_count: 5 }],
  activeSessionId: 's1',
  messages: [
    { id: '1', role: 'user', content: 'Hello', timestamp: 0 },
    { id: '2', role: 'assistant', content: 'Hi there', timestamp: 0 },
  ],
  currentSoul: { name: 'Friendly', description: 'A friendly soul', traits: ['warm'] },
  souls: [{ name: 'Friendly', description: 'A friendly soul', traits: ['warm'] }],
  switchSoul: jest.fn(),
  isConnected: true,
  chatBackground: '',
  updateTheme: jest.fn(),
  themeMode: 'dark',
  sessionLabels: { s1: ['test-label'] } as Record<string, string[]>,
  setSessionLabels: jest.fn(),
  allLabels: ['test-label'],
  setAllLabels: jest.fn(),
  labelInput: '',
  setLabelInput: jest.fn(),
  forwardMessage: jest.fn(async () => {}),
  createSession: jest.fn(async () => ({})),
  handleExportChat: jest.fn(),
  setShowSearch: jest.fn(),
}

describe('journey: Mobile: See What the AI Sees', () => {
  it('"+ Context" note reaches the /chat/control contract for the active chat', async () => {
    const { getByTestId, getByText, queryByTestId } = await render(<BarHarness />)

    expect(queryByTestId('context-input')).toBeNull()
    await fireEvent.press(getByTestId('inject-context-trigger'))
    await fireEvent.changeText(getByTestId('context-input'), 'I am writing a wedding speech')
    await fireEvent.press(getByTestId('context-send'))

    await waitFor(() =>
      expect(mockApi.post).toHaveBeenCalledWith('/chat/control', {
        session_id: 's1',
        action: 'context',
        context: 'I am writing a wedding speech',
      }),
    )
    await waitFor(() => expect(queryByTestId('context-input')).toBeNull())
    expect(getByText('+ Context')).toBeTruthy()
  })

  it('Details → Context Window shows what the AI holds, plain-language sections', async () => {
    const { getAllByText, getByText, queryByText } = await render(
      <ChatBottomSheets {...sheetProps} />,
    )

    await waitFor(() => expect(mockApi.inspectContext).toHaveBeenCalledTimes(1))
    await waitFor(() => expect(getByText('Context Window')).toBeTruthy())

    // Content is collapsed by default — expand all, then the sections show it.
    await fireEvent.press(getByText('Expand All'))
    expect(getAllByText('System Prompt').length).toBeGreaterThan(0)
    expect(getByText('hello there')).toBeTruthy()
    expect(getByText(/user_name=Ana/)).toBeTruthy()
    expect(queryByText('Loading context…')).toBeNull()
  })

  it('inspection failure degrades to an empty viewer, never a crash', async () => {
    ;(mockApi.inspectContext as jest.Mock).mockRejectedValue(new Error('boom'))
    const { getByText, queryByText } = await render(<ChatBottomSheets {...sheetProps} />)

    await waitFor(() => expect(mockApi.inspectContext).toHaveBeenCalledTimes(1))
    await waitFor(() => expect(getByText('No context items loaded')).toBeTruthy())
    expect(queryByText('boom')).toBeNull()
  })

  it('opening Details again re-reads the live state', async () => {
    const { rerender } = await render(<ChatBottomSheets {...sheetProps} showInfo={false} />)
    expect(mockApi.inspectContext).not.toHaveBeenCalled()
    await rerender(<ChatBottomSheets {...sheetProps} showInfo />)
    await waitFor(() => expect(mockApi.inspectContext).toHaveBeenCalled())
  })
})
