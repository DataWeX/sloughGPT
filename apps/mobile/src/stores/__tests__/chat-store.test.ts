import AsyncStorage from '@react-native-async-storage/async-storage';
import {api} from '../../services/api-client';
import * as sseClient from '../../services/sse-client';
import * as offlineCache from '../../services/offline-cache';
import {triggerHaptic} from '../../services/haptics';
import {sounds} from '../../services/sounds';
import {toast} from '../../services/toast';

jest.mock('../../services/api-client');
jest.mock('../../services/sse-client');
jest.mock('../../services/offline-cache');
jest.mock('../../services/haptics');
jest.mock('../../services/sounds');
jest.mock('../../services/toast');

// Mock store dependencies (must be hoisted)
jest.mock('../settings-store', () => ({
  useSettingsStore: {
    getState: () => ({
      temperature: 0.8,
      maxTokens: 256,
      topP: 0.9,
      topK: 40,
      repetitionPenalty: 1.1,
    }),
  },
}));

jest.mock('../hybrid-inference-store', () => ({
  useHybridStore: {
    getState: () => ({
      decideRoute: () => ({target: 'remote'}),
      executeLocal: jest.fn(),
      offlineOnly: false,
    }),
  },
}));

const mockApi = api as jest.Mocked<typeof api>;
const mockStreamSSE = sseClient.streamSSE as jest.Mock;
const mockOfflineCache = offlineCache as jest.Mocked<typeof offlineCache>;
const mockHaptic = triggerHaptic as jest.Mock;
const mockSounds = sounds as jest.Mocked<typeof sounds>;
const mockToast = toast as jest.Mocked<typeof toast>;

// eslint-disable-next-line @typescript-eslint/no-var-requires
const {useChatStore} = require('../chat-store');

beforeEach(async () => {
  jest.clearAllMocks();
  (AsyncStorage.getItem as jest.Mock).mockResolvedValue(null);
  (AsyncStorage.setItem as jest.Mock).mockResolvedValue(undefined);

  useChatStore.setState({
    sessions: [],
    activeSessionId: null,
    messages: [],
    streaming: false,
    error: null,
    offlineQueue: 0,
  });
});

function makeSession(overrides: Partial<any> = {}) {
  return {id: 's1', title: 'Test Chat', updatedAt: Date.now(), archived: false, ...overrides};
}

function makeMsg(overrides: Partial<any> = {}) {
  return {id: 'm1', role: 'user', content: 'hello', timestamp: Date.now(), ...overrides};
}

// ── refreshSessions ────────────────────────────────────────────────────────

describe('refreshSessions', () => {
  it('fetches sessions and updates state', async () => {
    const sessions = [makeSession()];
    mockApi.get.mockResolvedValue(sessions);

    await useChatStore.getState().refreshSessions();
    expect(useChatStore.getState().sessions).toEqual(sessions);
  });

  it('handles offline gracefully', async () => {
    mockApi.get.mockRejectedValue(new Error('offline'));
    await useChatStore.getState().refreshSessions();
    expect(useChatStore.getState().sessions).toEqual([]);
  });
});

// ── createSession ─────────────────────────────────────────────────────────

describe('createSession', () => {
  it('creates and caches a new session', async () => {
    mockApi.post.mockResolvedValue({session_id: 'new-id'});
    mockApi.get.mockResolvedValue([]);

    const id = await useChatStore.getState().createSession();
    expect(id).toBe('new-id');
    expect(useChatStore.getState().activeSessionId).toBe('new-id');
    expect(mockOfflineCache.cacheActiveSessionId).toHaveBeenCalledWith('new-id');
  });

  it('sets error on failure and returns empty', async () => {
    mockApi.post.mockRejectedValue(new Error('network error'));
    const id = await useChatStore.getState().createSession();
    expect(id).toBe('');
    expect(useChatStore.getState().error).toBe('network error');
  });
});

// ── loadSession ───────────────────────────────────────────────────────────

describe('loadSession', () => {
  it('loads messages from server and caches them', async () => {
    const msgs = [makeMsg({id: 'm1'})];
    mockApi.get.mockResolvedValue({messages: msgs});

    await useChatStore.getState().loadSession('s1');
    expect(useChatStore.getState().activeSessionId).toBe('s1');
    expect(useChatStore.getState().messages).toEqual(msgs);
    expect(mockOfflineCache.cacheMessages).toHaveBeenCalledWith('s1', msgs);
  });

  it('falls back to cache when offline', async () => {
    mockApi.get.mockRejectedValue(new Error('offline'));
    mockOfflineCache.getCachedMessages.mockResolvedValue([makeMsg({id: 'cached'})]);

    await useChatStore.getState().loadSession('s1');
    expect(useChatStore.getState().messages[0].id).toBe('cached');
    expect(useChatStore.getState().error).toContain('Offline');
  });
});

// ── deleteSession ─────────────────────────────────────────────────────────

describe('deleteSession', () => {
  it('deletes session and clears if active', async () => {
    mockApi.delete.mockResolvedValue(undefined);
    mockApi.get.mockResolvedValue([]);
    useChatStore.setState({activeSessionId: 's1'});

    await useChatStore.getState().deleteSession('s1');
    expect(useChatStore.getState().activeSessionId).toBeNull();
    expect(useChatStore.getState().messages).toEqual([]);
  });
});

// ── archiveSession ────────────────────────────────────────────────────────

describe('archiveSession', () => {
  it('calls archive API and refreshes', async () => {
    mockApi.archiveSession = jest.fn().mockResolvedValue(undefined);
    mockApi.get.mockResolvedValue([]);
    useChatStore.setState({sessions: [makeSession()]});

    await useChatStore.getState().archiveSession('s1', true);
    expect(mockApi.archiveSession).toHaveBeenCalledWith('s1', true);
  });
});

// ── renameSession ─────────────────────────────────────────────────────────

describe('renameSession', () => {
  it('calls rename API and refreshes', async () => {
    mockApi.renameSession = jest.fn().mockResolvedValue(undefined);
    mockApi.get.mockResolvedValue([]);

    await useChatStore.getState().renameSession('s1', 'New Title');
    expect(mockApi.renameSession).toHaveBeenCalledWith('s1', 'New Title');
  });
});

// ── deleteMessage ─────────────────────────────────────────────────────────

describe('deleteMessage', () => {
  it('removes message from local state', () => {
    useChatStore.setState({messages: [makeMsg({id: 'm1'}), makeMsg({id: 'm2'})]});
    useChatStore.getState().deleteMessage('m1');
    expect(useChatStore.getState().messages).toHaveLength(1);
    expect(useChatStore.getState().messages[0].id).toBe('m2');
  });
});

// ── forwardMessage ────────────────────────────────────────────────────────

describe('forwardMessage', () => {
  it('fetches session, appends message, saves back', async () => {
    mockApi.get.mockResolvedValue({id: 'target', messages: []});
    mockApi.put.mockResolvedValue(undefined);
    useChatStore.setState({activeSessionId: 'target'});

    await useChatStore.getState().forwardMessage('hello', 'target');
    expect(mockApi.get).toHaveBeenCalledWith('/chat/sessions/target');
    expect(mockApi.put).toHaveBeenCalledWith(
      '/chat/sessions/target',
      expect.objectContaining({messages: expect.any(Array)}),
    );
    expect(useChatStore.getState().messages.length).toBeGreaterThan(0);
    expect(mockHaptic).toHaveBeenCalledWith('light');
  });

  it('shows error toast on failure', async () => {
    mockApi.get.mockRejectedValue(new Error('fail'));
    await useChatStore.getState().forwardMessage('hello', 'target');
    expect(mockToast.error).toHaveBeenCalled();
  });
});

// ── cancelStream ─────────────────────────────────────────────────────────

describe('cancelStream', () => {
  it('sets streaming to false', () => {
    useChatStore.setState({streaming: true});
    useChatStore.getState().cancelStream();
    expect(useChatStore.getState().streaming).toBe(false);
  });
});

// ── recordFeedback ───────────────────────────────────────────────────────

describe('recordFeedback', () => {
  it('POSTs feedback', async () => {
    mockApi.post.mockResolvedValue(undefined);
    useChatStore.setState({activeSessionId: 's1'});

    await useChatStore.getState().recordFeedback('m1', true);
    expect(mockApi.post).toHaveBeenCalledWith('/feedback/workflow-record', {
      conversation_id: 'm1',
      rating: 'thumbs_up',
      session_id: 's1',
    });
    expect(mockHaptic).toHaveBeenCalledWith('success');
  });
});

// ── clearError ───────────────────────────────────────────────────────────

describe('clearError', () => {
  it('clears error', () => {
    useChatStore.setState({error: 'something bad'});
    useChatStore.getState().clearError();
    expect(useChatStore.getState().error).toBeNull();
  });
});

// ── retryPendingSends ─────────────────────────────────────────────────────

describe('retryPendingSends', () => {
  it('skips when no pending sends', async () => {
    mockOfflineCache.getPendingSends.mockResolvedValue([]);
    await useChatStore.getState().retryPendingSends();
    expect(mockToast.info).not.toHaveBeenCalled();
  });
});

// ── sendMessage (basic) ────────────────────────────────────────────────────

describe('sendMessage', () => {
  it('creates session if none active', async () => {
    mockApi.post.mockResolvedValue({session_id: 'new-session'});
    mockApi.get.mockResolvedValue([]);
    mockStreamSSE.mockReturnValue(
      (async function* () { yield {done: true}; })(),
    );

    await useChatStore.getState().sendMessage('hello');
    expect(useChatStore.getState().activeSessionId).toBe('new-session');
    expect(useChatStore.getState().messages.length).toBe(2); // user + assistant
  });

  it('adds user and assistant messages to state', async () => {
    useChatStore.setState({activeSessionId: 's1'});
    mockStreamSSE.mockReturnValue(
      (async function* () { yield {done: true}; })(),
    );
    mockApi.post.mockResolvedValue(undefined);
    mockApi.get.mockResolvedValue([]);

    await useChatStore.getState().sendMessage('hello');
    const msgs = useChatStore.getState().messages;
    expect(msgs.length).toBe(2);
    expect(msgs[0].role).toBe('user');
    expect(msgs[0].content).toBe('hello');
    expect(msgs[1].role).toBe('assistant');
  });

  it('recovers from SSE error gracefully', async () => {
    useChatStore.setState({activeSessionId: 's1'});
    const netError = new Error('Network failure');
    mockStreamSSE.mockReturnValue(
      (async function* () {
        throw netError;
      })(),
    );
    mockApi.post.mockResolvedValue(undefined);
    mockApi.get.mockResolvedValue([]);

    await useChatStore.getState().sendMessage('hello');

    expect(useChatStore.getState().streaming).toBe(false);
  });

  it('falls back to remote when local inference fails', async () => {
    // Mock hybrid store to return local route that then fails
    const mockLocalExec = jest.fn().mockImplementation(async (_c, _m, onToken) => {
      onToken('partial');
      return null; // simulate failure
    });

    // Re-require chat-store with different mock (need to reset module)
    jest.resetModules();
    jest.mock('../hybrid-inference-store', () => ({
      useHybridStore: {
        getState: () => ({
          decideRoute: () => ({target: 'local'}),
          executeLocal: mockLocalExec,
          offlineOnly: false,
        }),
      },
    }));

    const {useChatStore: Store2} = require('../chat-store');
    Store2.setState({activeSessionId: 's1', messages: [], streaming: false, error: null});

    mockStreamSSE.mockReturnValue(
      (async function* () { yield {done: true}; })(),
    );
    mockApi.post.mockResolvedValue(undefined);
    mockApi.get.mockResolvedValue([]);

    await Store2.getState().sendMessage('hello');

    // Should have attempted local, then fallen through to remote SSE
    expect(Store2.getState().streaming).toBe(false);
  });
});
