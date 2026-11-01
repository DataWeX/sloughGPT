import AsyncStorage from '@react-native-async-storage/async-storage';

const STORAGE_KEY = '@sloughgpt/knowledge';

// eslint-disable-next-line @typescript-eslint/no-var-requires
const {
  saveToKnowledge,
  getKnowledge,
  removeKnowledge,
  clearKnowledge,
  _resetCache,
} = require('../knowledge-store');

beforeEach(() => {
  jest.clearAllMocks();
  (AsyncStorage.getItem as jest.Mock).mockResolvedValue(null);
  (AsyncStorage.setItem as jest.Mock).mockResolvedValue(undefined);
  _resetCache();
});

describe('knowledge-store', () => {
  describe('saveToKnowledge', () => {
    it('saves a new fact', async () => {
      const result = await saveToKnowledge('Test content', 'user', 'msg-1');
      expect(result).toBe(true);
      expect(AsyncStorage.setItem).toHaveBeenCalledWith(
        STORAGE_KEY,
        expect.stringContaining('Test content'),
      );
    });

    it('deduplicates by messageId', async () => {
      const existing = [{id: '1', content: 'Old', source: 'user', messageId: 'msg-1', savedAt: 100}];
      (AsyncStorage.getItem as jest.Mock).mockResolvedValue(JSON.stringify(existing));

      const result = await saveToKnowledge('Duplicate', 'user', 'msg-1');
      expect(result).toBe(false);
    });

    it('adds new fact at the beginning', async () => {
      const existing = [{id: '1', content: 'Old', source: 'user', messageId: 'msg-old', savedAt: 100}];
      (AsyncStorage.getItem as jest.Mock).mockResolvedValue(JSON.stringify(existing));

      await saveToKnowledge('New fact', 'assistant', 'msg-new');
      const saved = JSON.parse((AsyncStorage.setItem as jest.Mock).mock.calls[0][1]);
      expect(saved[0].content).toBe('New fact');
      expect(saved[0].source).toBe('assistant');
      expect(saved[1].content).toBe('Old');
    });

    it('generates unique ids', async () => {
      await saveToKnowledge('Fact 1', 'user', 'msg-1');
      await saveToKnowledge('Fact 2', 'user', 'msg-2');
      const call1 = JSON.parse((AsyncStorage.setItem as jest.Mock).mock.calls[0][1]);
      const call2 = JSON.parse((AsyncStorage.setItem as jest.Mock).mock.calls[1][1]);
      expect(call1[0].id).not.toBe(call2[0].id);
    });
  });

  describe('getKnowledge', () => {
    it('returns empty array when nothing stored', async () => {
      const result = await getKnowledge();
      expect(result).toEqual([]);
    });

    it('returns stored facts', async () => {
      const facts = [{id: '1', content: 'Test', source: 'user', messageId: 'msg-1', savedAt: 100}];
      (AsyncStorage.getItem as jest.Mock).mockResolvedValue(JSON.stringify(facts));

      const result = await getKnowledge();
      expect(result).toHaveLength(1);
      expect(result[0].content).toBe('Test');
    });

    it('uses cache on subsequent calls', async () => {
      const facts = [{id: '1', content: 'Cached', source: 'user', messageId: 'msg-1', savedAt: 100}];
      (AsyncStorage.getItem as jest.Mock).mockResolvedValue(JSON.stringify(facts));

      await getKnowledge();
      await getKnowledge();
      expect(AsyncStorage.getItem).toHaveBeenCalledTimes(1);
    });
  });

  describe('removeKnowledge', () => {
    it('removes fact by id', async () => {
      const facts = [
        {id: '1', content: 'Keep', source: 'user', messageId: 'msg-1', savedAt: 100},
        {id: '2', content: 'Remove', source: 'user', messageId: 'msg-2', savedAt: 200},
      ];
      (AsyncStorage.getItem as jest.Mock).mockResolvedValue(JSON.stringify(facts));

      await removeKnowledge('2');
      const saved = JSON.parse((AsyncStorage.setItem as jest.Mock).mock.calls[0][1]);
      expect(saved).toHaveLength(1);
      expect(saved[0].id).toBe('1');
    });
  });

  describe('clearKnowledge', () => {
    it('clears all facts', async () => {
      const facts = [{id: '1', content: 'Test', source: 'user', messageId: 'msg-1', savedAt: 100}];
      (AsyncStorage.getItem as jest.Mock).mockResolvedValue(JSON.stringify(facts));

      await clearKnowledge();
      expect(AsyncStorage.removeItem).toHaveBeenCalledWith(STORAGE_KEY);
    });
  });
});
