import {useChatStore} from '../chat-store';

const mockCreateSession = jest.fn();
const mockTriggerHaptic = jest.fn();
const mockSoundsSend = jest.fn();
const mockSoundsReceive = jest.fn();
let mockStreamSSE: jest.Mock;

jest.mock('../../services/sse-client', () => ({
  streamSSE: jest.fn((..._args: any[]) => mockStreamSSE()),
}));

jest.mock('../hybrid-inference-store', () => {
  const _decide = jest.fn().mockReturnValue({target: 'remote', reason: 'no local engine loaded'});
  const _exec = jest.fn();
  return {
    useHybridStore: { getState: () => ({ decideRoute: _decide, executeLocal: _exec }) },
    __mockRoute: { _decide, _exec },
  };
});

jest.mock('../../services/offline-cache', () => ({
  cacheMessages: jest.fn(),
  getCachedMessages: jest.fn().mockResolvedValue([]),
  appendCachedMessage: jest.fn(),
  addPendingSend: jest.fn(),
  getPendingSends: jest.fn().mockResolvedValue([]),
  removePendingSend: jest.fn(),
  cacheActiveSessionId: jest.fn(),
}));

jest.mock('../../services/haptics', () => ({
  triggerHaptic: (...args: any[]) => mockTriggerHaptic(...args),
}));

jest.mock('../../services/sounds', () => ({
  sounds: { send: (...a: any[]) => mockSoundsSend(...a), receive: (...a: any[]) => mockSoundsReceive(...a), error: jest.fn() },
}));

jest.mock('../../services/toast', () => ({
  toast: { success: jest.fn(), error: jest.fn(), warn: jest.fn(), info: jest.fn() },
}));

jest.mock('../settings-store', () => ({
  useSettingsStore: { getState: () => ({ temperature: 0.8, maxTokens: 256, topP: 0.9, topK: 50, repetitionPenalty: 1.2 }) },
}));

jest.mock('../../services/api-client', () => ({
  api: {
    get: jest.fn(),
    post: jest.fn((url: string) => url === '/chat/sessions' ? (mockCreateSession(), Promise.resolve({id: 's-1'})) : Promise.resolve({})),
    delete: jest.fn(), archiveSession: jest.fn(), renameSession: jest.fn(),
  },
}));

function mockRoute() { return require('../hybrid-inference-store').__mockRoute; }

beforeEach(() => {
  jest.clearAllMocks();
  mockRoute()._decide.mockReturnValue({target: 'remote', reason: 'no local engine loaded'});
  mockStreamSSE = jest.fn().mockImplementation(async function* () { yield {done: true}; });
  useChatStore.setState({ sessions: [], activeSessionId: null, messages: [], streaming: false, error: null, offlineQueue: 0 });
});

describe('sendMessage — local slonet', () => {
  it('updates assistant content from executeLocal', async () => {
    mockRoute()._decide.mockReturnValue({target: 'local', engine: 'slonet'});
    mockRoute()._exec.mockResolvedValue({text: 'device reply', tokens_generated: 2, elapsed_ms: 10});

    await useChatStore.getState().sendMessage('hello');

    const state = useChatStore.getState();
    expect(mockRoute()._exec).toHaveBeenCalled();
    const assistant = state.messages.find(m => m.role === 'assistant');
    expect(assistant?.content).toBe('device reply');
    expect(state.streaming).toBe(false);
  });
});

describe('sendMessage — local qwen', () => {
  it('updates assistant content from executeLocal', async () => {
    mockRoute()._decide.mockReturnValue({target: 'local', engine: 'qwen'});
    mockRoute()._exec.mockResolvedValue({text: 'qwen local', tokens_generated: 3, elapsed_ms: 200});

    await useChatStore.getState().sendMessage('test');

    const state = useChatStore.getState();
    const assistant = state.messages.find(m => m.role === 'assistant');
    expect(assistant?.content).toBe('qwen local');
    expect(state.streaming).toBe(false);
  });
});

describe('sendMessage — fallback to remote', () => {
  it('falls through to remote when executeLocal returns null', async () => {
    mockRoute()._decide.mockReturnValue({target: 'local', engine: 'slonet'});
    mockRoute()._exec.mockResolvedValue(null);
    mockStreamSSE = jest.fn().mockImplementation(async function* () {
      yield {token: 'server ', done: false};
      yield {token: 'reply', done: true};
    });

    await useChatStore.getState().sendMessage('hi');

    const state = useChatStore.getState();
    const assistant = state.messages.find(m => m.role === 'assistant');
    expect(assistant?.content).toBe('server reply');
  });

  it('falls through when executeLocal throws', async () => {
    mockRoute()._decide.mockReturnValue({target: 'local', engine: 'qwen'});
    mockRoute()._exec.mockRejectedValue(new Error('fail'));
    mockStreamSSE = jest.fn().mockImplementation(async function* () {
      yield {token: 'fallback', done: true};
    });

    await useChatStore.getState().sendMessage('x');

    const state = useChatStore.getState();
    const assistant = state.messages.find(m => m.role === 'assistant');
    expect(assistant?.content).toBe('fallback');
  });

  it('defaults to remote when no local engine', async () => {
    await useChatStore.getState().sendMessage('yo');

    expect(mockRoute()._exec).not.toHaveBeenCalled();
    expect(useChatStore.getState().streaming).toBe(false);
  });
});

describe('regenerate — local inference', () => {
  it('regenerates via local engine', async () => {
    mockRoute()._decide.mockReturnValue({target: 'local', engine: 'slonet'});
    mockRoute()._exec.mockResolvedValue({text: 'first', tokens_generated: 1, elapsed_ms: 5});

    await useChatStore.getState().sendMessage('hello');
    const msgs = useChatStore.getState().messages;
    const assistant = msgs.find(m => m.role === 'assistant');
    expect(assistant?.content).toBe('first');

    // Set up regenerate to use qwen locally
    mockRoute()._decide.mockReturnValue({target: 'local', engine: 'qwen'});
    mockRoute()._exec.mockResolvedValue({text: 'regenerated', tokens_generated: 2, elapsed_ms: 50});

    await useChatStore.getState().regenerate(assistant!.id);

    const updated = useChatStore.getState().messages;
    const updatedAssistant = updated.find(m => m.role === 'assistant');
    expect(updatedAssistant?.content).toBe('regenerated');
  });

  it('falls back to remote when local regenerate fails', async () => {
    mockRoute()._decide.mockReturnValue({target: 'local', engine: 'slonet'});
    mockRoute()._exec.mockResolvedValue({text: 'first', tokens_generated: 1, elapsed_ms: 5});

    await useChatStore.getState().sendMessage('hello');
    const msgId = useChatStore.getState().messages.find(m => m.role === 'assistant')!.id;

    mockRoute()._decide.mockReturnValue({target: 'local', engine: 'qwen'});
    mockRoute()._exec.mockRejectedValue(new Error('fail'));
    mockStreamSSE = jest.fn().mockImplementation(async function* () {
      yield {token: 'remote ', done: false};
      yield {token: 'regeneration', done: true};
    });

    await useChatStore.getState().regenerate(msgId);

    const updated = useChatStore.getState().messages;
    expect(updated.find(m => m.role === 'assistant')?.content).toBe('remote regeneration');
  });
});
