import AsyncStorage from '@react-native-async-storage/async-storage';
import {
  cacheMessages,
  getCachedMessages,
  appendCachedMessage,
  removeCachedMessage,
  addPendingSend,
  getPendingSends,
  removePendingSend,
  incrementPendingRetry,
  cacheActiveSessionId,
  getCachedActiveSessionId,
} from '../offline-cache';

beforeEach(async () => {
  jest.clearAllMocks();
  (AsyncStorage.getItem as jest.Mock).mockResolvedValue(null);
  (AsyncStorage.setItem as jest.Mock).mockResolvedValue(undefined);
  (AsyncStorage.removeItem as jest.Mock).mockResolvedValue(undefined);
});

function msg(overrides: Partial<any> = {}) {
  return {id: 'm1', role: 'user', content: 'hi', timestamp: Date.now(), ...overrides};
}

describe('cacheMessages', () => {
  it('stores messages by session', async () => {
    await cacheMessages('s1', [msg()]);
    expect(AsyncStorage.setItem).toHaveBeenCalledWith(
      '@sloughgpt/messages',
      expect.stringContaining('"s1"'),
    );
  });

  it('merges with existing cached data', async () => {
    (AsyncStorage.getItem as jest.Mock).mockResolvedValue(
      JSON.stringify({s1: [msg({id: 'old'})]}),
    );
    await cacheMessages('s1', [msg({id: 'new'})]);
    const saved = JSON.parse((AsyncStorage.setItem as jest.Mock).mock.calls[0][1]);
    expect(saved.s1).toHaveLength(1);
    expect(saved.s1[0].id).toBe('new'); // replaced
  });
});

describe('getCachedMessages', () => {
  it('returns empty array when nothing cached', async () => {
    const msgs = await getCachedMessages('s1');
    expect(msgs).toEqual([]);
  });

  it('returns cached messages for session', async () => {
    (AsyncStorage.getItem as jest.Mock).mockResolvedValue(
      JSON.stringify({s1: [msg({id: 'cached'})]}),
    );
    const msgs = await getCachedMessages('s1');
    expect(msgs).toHaveLength(1);
    expect(msgs[0].id).toBe('cached');
  });

  it('returns empty for unknown session', async () => {
    (AsyncStorage.getItem as jest.Mock).mockResolvedValue(
      JSON.stringify({s1: [msg()]}),
    );
    const msgs = await getCachedMessages('s2');
    expect(msgs).toEqual([]);
  });
});

describe('appendCachedMessage', () => {
  it('appends to existing session', async () => {
    (AsyncStorage.getItem as jest.Mock).mockResolvedValue(
      JSON.stringify({s1: [msg({id: 'm1'})]}),
    );
    await appendCachedMessage('s1', msg({id: 'm2'}));
    const saved = JSON.parse((AsyncStorage.setItem as jest.Mock).mock.calls[0][1]);
    expect(saved.s1).toHaveLength(2);
    expect(saved.s1[0].id).toBe('m1');
    expect(saved.s1[1].id).toBe('m2');
  });

  it('creates new entry for new session', async () => {
    await appendCachedMessage('s1', msg({id: 'm1'}));
    const saved = JSON.parse((AsyncStorage.setItem as jest.Mock).mock.calls[0][1]);
    expect(saved.s1).toHaveLength(1);
    expect(saved.s1[0].id).toBe('m1');
  });
});

describe('removeCachedMessage', () => {
  it('removes message by id from session', async () => {
    (AsyncStorage.getItem as jest.Mock).mockResolvedValue(
      JSON.stringify({s1: [msg({id: 'm1'}), msg({id: 'm2'})]}),
    );
    await removeCachedMessage('s1', 'm1');
    const saved = JSON.parse((AsyncStorage.setItem as jest.Mock).mock.calls[0][1]);
    expect(saved.s1).toHaveLength(1);
    expect(saved.s1[0].id).toBe('m2');
  });

  it('handles non-existent id gracefully', async () => {
    (AsyncStorage.getItem as jest.Mock).mockResolvedValue(
      JSON.stringify({s1: [msg({id: 'm1'})]}),
    );
    await removeCachedMessage('s1', 'nonexistent');
    const saved = JSON.parse((AsyncStorage.setItem as jest.Mock).mock.calls[0][1]);
    expect(saved.s1).toHaveLength(1);
  });
});

describe('pending sends', () => {
  const pending = {id: 'p1', sessionId: 's1', content: 'hi', timestamp: 100, retryCount: 0};

  it('addPendingSend stores a pending message', async () => {
    await addPendingSend(pending);
    expect(AsyncStorage.setItem).toHaveBeenCalledWith(
      '@sloughgpt/pending_sends',
      expect.stringContaining('"p1"'),
    );
  });

  it('getPendingSends returns empty when nothing stored', async () => {
    const result = await getPendingSends();
    expect(result).toEqual([]);
  });

  it('getPendingSends returns stored items', async () => {
    (AsyncStorage.getItem as jest.Mock).mockResolvedValue(JSON.stringify([pending]));
    const result = await getPendingSends();
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe('p1');
  });

  it('removePendingSend removes by id', async () => {
    (AsyncStorage.getItem as jest.Mock).mockResolvedValue(
      JSON.stringify([pending, {id: 'p2', sessionId: 's1', content: 'bye', timestamp: 200, retryCount: 0}]),
    );
    await removePendingSend('p1');
    const saved = JSON.parse((AsyncStorage.setItem as jest.Mock).mock.calls[0][1]);
    expect(saved).toHaveLength(1);
    expect(saved[0].id).toBe('p2');
  });

  it('incrementPendingRetry increments retryCount', async () => {
    (AsyncStorage.getItem as jest.Mock).mockResolvedValue(
      JSON.stringify([{...pending, retryCount: 0}]),
    );
    await incrementPendingRetry('p1');
    const saved = JSON.parse((AsyncStorage.setItem as jest.Mock).mock.calls[0][1]);
    expect(saved[0].retryCount).toBe(1);
  });
});

describe('active session id', () => {
  it('cacheActiveSessionId stores the id', async () => {
    await cacheActiveSessionId('s1');
    expect(AsyncStorage.setItem).toHaveBeenCalledWith('@sloughgpt/active_session', 's1');
  });

  it('cacheActiveSessionId removes when null', async () => {
    await cacheActiveSessionId(null);
    expect(AsyncStorage.removeItem).toHaveBeenCalledWith('@sloughgpt/active_session');
  });

  it('getCachedActiveSessionId returns stored id', async () => {
    (AsyncStorage.getItem as jest.Mock).mockResolvedValue('s1');
    const id = await getCachedActiveSessionId();
    expect(id).toBe('s1');
  });

  it('getCachedActiveSessionId returns null when empty', async () => {
    const id = await getCachedActiveSessionId();
    expect(id).toBeNull();
  });
});
