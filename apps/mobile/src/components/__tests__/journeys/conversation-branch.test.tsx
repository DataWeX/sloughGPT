/**
 * Journey: Mobile: Branch a Conversation (UX_FLOWS.md)
 *
 * In-code journey: long-press a message → "Branch from here" drives the
 * real chat-store into the branch contract, and the Details sheet shows the
 * real lineage payload. Only transports/services are mocked. Mapped in
 * journeys/flows.py as m3-branch.
 */
import React from 'react'
import AsyncStorage from '@react-native-async-storage/async-storage'
import { fireEvent, render, waitFor } from '../../../test-utils'
import { MessageBubble } from '../../MessageBubble'
import { ChatBottomSheets } from '../../ChatBottomSheets'
import { api } from '../../../services/api-client'
import { useChatStore } from '../../../stores/chat-store'

jest.mock('../../../services/api-client')
jest.mock('../../../services/sse-client')
jest.mock('../../../services/offline-cache')
jest.mock('../../../services/haptics')
jest.mock('../../../services/sounds')
jest.mock('../../../services/toast')
jest.mock('../../../services/clipboard', () => ({ copyToClipboard: jest.fn() }))
jest.mock('../../../services/bookmarks', () => ({
  addBookmark: jest.fn(),
  removeBookmark: jest.fn(),
  isBookmarked: jest.fn(() => Promise.resolve(false)),
}))
jest.mock('../../../services/pins', () => ({
  pinMessage: jest.fn(),
  unpinMessage: jest.fn(),
  isPinned: jest.fn(() => Promise.resolve(false)),
}))
jest.mock('../../../services/reactions', () => ({
  getMessageReactions: jest.fn(() => Promise.resolve({})),
  toggleReaction: jest.fn(),
  REACTION_EMOJIS: [],
}))
jest.mock('../../../services/knowledge-store', () => ({
  saveToKnowledge: jest.fn(),
  getKnowledgeForMessage: jest.fn(() => Promise.resolve(null)),
}))
jest.mock('../../../stores/settings-store', () => {
  const state = {
    theme: 'dark',
    temperature: 0.8,
    maxTokens: 256,
    topP: 0.9,
    topK: 40,
    repetitionPenalty: 1.1,
    chatBackground: '',
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

const SRC_MESSAGES = [
  { id: 'm0', role: 'user', content: 'first question', timestamp: 1 },
  { id: 'm1', role: 'assistant', content: 'first answer', timestamp: 2 },
  { id: 'm2', role: 'user', content: 'second question', timestamp: 3 },
]

const LINEAGE_FIXTURE = {
  session_id: 'new9',
  parent_session_id: 'src1',
  branch_from_index: 1,
  ancestors: [
    { session_id: 'src1', parent_session_id: null, branch_from_index: null, name: 'Original chat' },
  ],
}

beforeEach(() => {
  jest.clearAllMocks()
  const store = new Map<string, string>()
  jest.spyOn(AsyncStorage, 'getItem').mockImplementation(async (key) => store.get(key) ?? null)
  jest.spyOn(AsyncStorage, 'setItem').mockImplementation(async (key, value) => {
    store.set(key, String(value))
  })
  ;(mockApi.get as jest.Mock).mockImplementation(async (path: string) => {
    if (path === '/session/src1/messages') {
      return { messages: SRC_MESSAGES }
    }
    if (path === '/session/new9/messages') {
      return { messages: SRC_MESSAGES.slice(0, 2) }
    }
    if (path === '/chat/sessions') {
      return [
        { id: 'src1', name: 'Original chat', created_at: '', updated_at: '', message_count: 3 },
        {
          id: 'new9',
          name: 'Original chat (Branch)',
          created_at: '',
          updated_at: '',
          message_count: 2,
        },
      ]
    }
    return null
  })
  ;(mockApi.branchSession as jest.Mock).mockResolvedValue({
    session_id: 'new9',
    branched_from: 'src1',
    from_index: 1,
  })
  ;(mockApi.getSessionLineage as jest.Mock).mockResolvedValue(LINEAGE_FIXTURE)
  useChatStore.setState({
    activeSessionId: 'src1',
    messages: SRC_MESSAGES,
    sessions: [],
  })
})

function BubbleHarness({ index }: { index: number }) {
  const branchFromMessage = useChatStore((s: any) => s.branchFromMessage)
  return (
    <MessageBubble
      message={SRC_MESSAGES[index]}
      sessionId="src1"
      onBranch={() => {
        branchFromMessage(index)
      }}
    />
  )
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
  safeSessions: [
    {
      id: 'new9',
      name: 'Original chat (Branch)',
      created_at: '',
      updated_at: '',
      message_count: 2,
    },
  ],
  activeSessionId: 'new9',
  messages: SRC_MESSAGES.slice(0, 2).map((m) => ({ ...m, timestamp: 0 })),
  currentSoul: { name: 'Friendly', description: 'A friendly soul', traits: ['warm'] },
  souls: [{ name: 'Friendly', description: 'A friendly soul', traits: ['warm'] }],
  switchSoul: jest.fn(),
  isConnected: true,
  chatBackground: '',
  updateTheme: jest.fn(),
  themeMode: 'dark',
  sessionLabels: {} as Record<string, string[]>,
  setSessionLabels: jest.fn(),
  allLabels: [] as string[],
  setAllLabels: jest.fn(),
  labelInput: '',
  setLabelInput: jest.fn(),
  forwardMessage: jest.fn(async () => {}),
  createSession: jest.fn(async () => ({})),
  handleExportChat: jest.fn(),
  setShowSearch: jest.fn(),
}

describe('journey: Mobile: Branch a Conversation', () => {
  it('long-press → "Branch from here" forks the chat and opens the branch', async () => {
    const { getByText } = await render(<BubbleHarness index={1} />)

    await fireEvent(getByText('first answer'), 'longPress')
    await waitFor(() => expect(getByText('Branch from here')).toBeTruthy())

    await fireEvent.press(getByText('Branch from here'))

    await waitFor(() => expect(mockApi.branchSession).toHaveBeenCalledWith('src1', 1))
    // branch history (prefix through message #2) loaded into the active chat
    await waitFor(() => expect(useChatStore.getState().activeSessionId).toBe('new9'))
    expect(useChatStore.getState().messages).toEqual(SRC_MESSAGES.slice(0, 2))
    // drawer refreshed with the branch listed
    expect(useChatStore.getState().sessions.map((s) => s.id)).toEqual(['src1', 'new9'])
    expect(mockApi.delete).not.toHaveBeenCalled()
    expect(mockApi.put).not.toHaveBeenCalled()
  })

  it('Details → Branch Lineage shows where this chat came from', async () => {
    const { getByText } = await render(<ChatBottomSheets {...sheetProps} />)

    await waitFor(() => expect(mockApi.getSessionLineage).toHaveBeenCalledWith('new9'))
    await waitFor(() => expect(getByText('BRANCH LINEAGE')).toBeTruthy())
    expect(getByText(/This session · from message #2/)).toBeTruthy()
    expect(getByText(/Original chat/)).toBeTruthy()
  })

  it('lineage failure degrades quietly — no lineage section, no crash', async () => {
    ;(mockApi.getSessionLineage as jest.Mock).mockRejectedValue(new Error('boom'))
    const { queryByText } = await render(<ChatBottomSheets {...sheetProps} />)

    await waitFor(() => expect(mockApi.getSessionLineage).toHaveBeenCalledWith('new9'))
    expect(queryByText('BRANCH LINEAGE')).toBeNull()
    expect(queryByText('boom')).toBeNull()
  })
})
