/**
 * Journey: Mobile: Annotate a Message (UX_FLOWS.md)
 *
 * Slice 4 foundation + feature: message ids must survive reloads (the context
 * seam strips them today), and notes key off those ids. Only transports and
 * services are mocked — the real message-notes service runs against a mocked
 * AsyncStorage. Mapped in domain/journeys/flows.py as m4-notes.
 */
import React from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';
import {fireEvent, render, waitFor} from '../../../test-utils';
import {MessageBubble} from '../../MessageBubble';
import {api} from '../../../services/api-client';
import {useChatStore} from '../../../stores/chat-store';
import {getMessageNotes, saveMessageNote} from '../../../services/message-notes';

jest.mock('../../../services/api-client');
jest.mock('../../../services/sse-client');
jest.mock('../../../services/offline-cache');
jest.mock('../../../services/haptics');
jest.mock('../../../services/sounds');
jest.mock('../../../services/toast');
jest.mock('../../../services/clipboard', () => ({copyToClipboard: jest.fn()}));
jest.mock('../../../services/bookmarks', () => ({
  addBookmark: jest.fn(),
  removeBookmark: jest.fn(),
  isBookmarked: jest.fn(() => Promise.resolve(false)),
}));
jest.mock('../../../services/pins', () => ({
  pinMessage: jest.fn(),
  unpinMessage: jest.fn(),
  isPinned: jest.fn(() => Promise.resolve(false)),
}));
jest.mock('../../../services/reactions', () => ({
  getMessageReactions: jest.fn(() => Promise.resolve({})),
  toggleReaction: jest.fn(),
  REACTION_EMOJIS: [],
}));
jest.mock('../../../services/knowledge-store', () => ({
  saveToKnowledge: jest.fn(),
  getKnowledgeForMessage: jest.fn(() => Promise.resolve(null)),
}));
jest.mock('../../../stores/settings-store', () => {
  const state = {
    theme: 'dark',
    temperature: 0.8,
    maxTokens: 256,
    topP: 0.9,
    topK: 40,
    repetitionPenalty: 1.1,
    chatBackground: '',
  };
  const useSettingsStore: any = (selector: (s: typeof state) => unknown) => selector(state);
  useSettingsStore.getState = () => state;
  useSettingsStore.setState = () => state;
  useSettingsStore.subscribe = () => () => {};
  return {useSettingsStore, __esModule: true};
});
jest.mock('../../../stores/hybrid-inference-store', () => ({
  useHybridStore: {
    getState: () => ({
      decideRoute: () => ({target: 'remote'}),
      executeLocal: jest.fn(),
      offlineOnly: false,
    }),
  },
}));

const mockApi = api as jest.Mocked<typeof api>;

const RAW_MESSAGES = [
  {role: 'user', content: 'what did the fox say'},
  {role: 'assistant', content: 'something fox-like'},
];

let store = new Map<string, string>();

beforeEach(() => {
  jest.clearAllMocks();
  store = new Map();
  jest.spyOn(AsyncStorage, 'getItem').mockImplementation(async key => store.get(key) ?? null);
  jest.spyOn(AsyncStorage, 'setItem').mockImplementation(async (key, value) => {
    store.set(key, String(value));
  });
  (mockApi.get as jest.Mock).mockImplementation(async (path: string) => {
    if (path.startsWith('/session/') && path.endsWith('/messages')) {
      // legacy context store: role/content only, exactly what production
      // returned before the id passthrough landed
      return {messages: RAW_MESSAGES.map(m => ({...m}))};
    }
    return {messages: []};
  });
  useChatStore.setState({activeSessionId: 's1', messages: [], sessions: []});
});

describe('journey: Mobile: Annotate a Message', () => {
  it('loadSession fabricates stable ids so notes rebind across reloads', async () => {
    await useChatStore.getState().loadSession('s1');
    const first = useChatStore.getState().messages;
    expect(first).toHaveLength(2);
    expect(first[0].id).toBeTruthy();
    expect(first[1].id).toBeTruthy();
    expect(first[0].id).not.toBe(first[1].id);

    // a note saved against the loaded message, through the real service
    const noteId = first[0].id;
    await saveMessageNote({
      id: `s1:${noteId}`,
      sessionId: 's1',
      messageId: noteId,
      content: 'the fox bit line',
      createdAt: 1,
      updatedAt: 1,
    });

    // reload — same server payload, no ids — the id must be deterministic
    await useChatStore.getState().loadSession('s1');
    const second = useChatStore.getState().messages;
    expect(second[0].id).toBe(noteId);

    const notes = await getMessageNotes('s1');
    expect(notes.find(n => n.messageId === noteId)?.content).toBe('the fox bit line');
  });

  it('long-press → Add note → saved note renders as a strip on the bubble', async () => {
    // own session id — notes persist across tests, ids are deterministic
    await useChatStore.getState().loadSession('s2');
    const [msg] = useChatStore.getState().messages;
    const {getByText, getAllByText, getByPlaceholderText, queryByText} = await render(
      <MessageBubble message={msg} sessionId="s2" />,
    );

    // [0]: Markdown may split text nodes — target the bubble occurrence
    const hits = getAllByText('what did the fox say');
    await fireEvent(hits[0], 'longPress');
    await waitFor(() => expect(getByText('Add note')).toBeTruthy());
    await fireEvent.press(getByText('Add note'));

    await waitFor(() =>
      expect(getByPlaceholderText('Add a note...')).toBeTruthy(),
    );
    await fireEvent.changeText(getByPlaceholderText('Add a note...'), 'pun intended');
    await fireEvent.press(getByText('Save'));

    await waitFor(() => expect(getByText('pun intended')).toBeTruthy());
    expect(queryByText('Add a note...')).toBeNull();

    // note persisted through the real service against the stable id
    const notes = await getMessageNotes('s2');
    expect(notes.map(n => n.content)).toContain('pun intended');
  });

  it('clone persists message ids into the context write', async () => {
    await useChatStore.getState().loadSession('s1');
    const sourceId = useChatStore.getState().messages[0].id;

    let contextPayload: any = null;
    (mockApi.post as jest.Mock).mockImplementation(async (path: string, body: any) => {
      if (path === '/chat/sessions') {
        return {session_id: 'clone1'};
      }
      if (path === '/session/clone1/context') {
        contextPayload = body;
        return {status: 'stored'};
      }
      return {};
    });
    (mockApi.put as jest.Mock).mockResolvedValue({});

    await useChatStore.getState().cloneSession('s1', 'S1 (Copy)');

    expect(contextPayload).toBeTruthy();
    expect(contextPayload.messages).toHaveLength(2);
    expect(contextPayload.messages[0].id).toBe(`clone1-${sourceId}`);
    expect(contextPayload.messages[0].role).toBe('user');
  });
});
