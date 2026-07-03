import AsyncStorage from '@react-native-async-storage/async-storage';

// Mock api-client
jest.mock('../../services/api-client', () => ({
  getApiUrl: jest.fn(async () => 'http://localhost:8000'),
}));

// Mock activity-inference so initInference() is a no-op
jest.mock('../../services/activity-inference', () => ({
  initInference: jest.fn(async () => true),
}));

// Mock global fetch
const mockFetch = jest.fn();
global.fetch = mockFetch as any;

// Mock FileReader for blob->base64
class MockFileReader {
  result: string | null = null;
  onload: (() => void) | null = null;
  onerror: ((e: any) => void) | null = null;
  readAsDataURL(_blob: any) {
    this.result = 'data:application/octet-stream;base64,dGVzdA==';
    setTimeout(() => this.onload?.(), 0);
  }
}
(global as any).FileReader = MockFileReader;

beforeEach(async () => {
  jest.restoreAllMocks();
  (AsyncStorage as any)._store = {};
});

const mod = require('../model-sync');

describe('model-sync', () => {
  describe('getModelSyncStatus', () => {
    it('returns not cached when no data', async () => {
      const status = await mod.getModelSyncStatus();
      expect(status.cached).toBe(false);
      expect(status.version).toBe(0);
      expect(status.lastSync).toBeNull();
      expect(status.fileSize).toBeNull();
    });

    it('returns cached when flag is true', async () => {
      await AsyncStorage.setItem('@sloughgpt/activity_model_cached', 'true');
      await AsyncStorage.setItem('@sloughgpt/activity_model_version', '12345');

      const status = await mod.getModelSyncStatus();
      expect(status.cached).toBe(true);
      expect(status.version).toBe(12345);
      expect(status.lastSync).toBe(12345);
    });

    it('returns cached when data URL stored', async () => {
      const dataUrl = 'data:application/octet-stream;base64,dGVzdA==';
      await AsyncStorage.setItem('@sloughgpt/activity_model_cached', dataUrl);

      const status = await mod.getModelSyncStatus();
      expect(status.cached).toBe(true);
      expect(status.fileSize).toBe(dataUrl.length);
    });
  });

  describe('syncModel', () => {
    it('returns false when model not loaded on server', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({model_loaded: false}),
      });

      const result = await mod.syncModel();
      expect(result).toBe(false);
    });

    it('returns false when status endpoint fails', async () => {
      mockFetch.mockResolvedValueOnce({ok: false});

      const result = await mod.syncModel();
      expect(result).toBe(false);
    });

    it('downloads model and caches it', async () => {
      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({model_loaded: true}),
        })
        .mockResolvedValueOnce({
          ok: true,
          blob: async () => new Blob(['test']),
        });

      const result = await mod.syncModel();
      expect(result).toBe(true);

      const cached = await AsyncStorage.getItem('@sloughgpt/activity_model_cached');
      expect(cached).toBeTruthy();
    });

    it('returns false when download fails', async () => {
      // Ensure clean state — previous test's FileReader async callback
      // may have stored data after beforeEach cleared AsyncStorage.
      (AsyncStorage as any)._store = {};

      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({model_loaded: true}),
        })
        .mockResolvedValueOnce({ok: false});

      const result = await mod.syncModel();
      expect(result).toBe(false);

      const cached = await AsyncStorage.getItem('@sloughgpt/activity_model_cached');
      expect(cached).toBeNull();
    });
  });

  describe('clearCachedModel', () => {
    it('removes cached model data', async () => {
      await AsyncStorage.setItem('@sloughgpt/activity_model_cached', 'true');
      await AsyncStorage.setItem('@sloughgpt/activity_model_version', '123');

      await mod.clearCachedModel();

      const cached = await AsyncStorage.getItem('@sloughgpt/activity_model_cached');
      const version = await AsyncStorage.getItem('@sloughgpt/activity_model_version');
      expect(cached).toBeNull();
      expect(version).toBeNull();
    });
  });

  describe('getCachedModelData', () => {
    it('returns null when no cached model', async () => {
      const data = await mod.getCachedModelData();
      expect(data).toBeNull();
    });

    it('returns cached data URL', async () => {
      const dataUrl = 'data:application/octet-stream;base64,dGVzdA==';
      await AsyncStorage.setItem('@sloughgpt/activity_model_cached', dataUrl);

      const data = await mod.getCachedModelData();
      expect(data).toBe(dataUrl);
    });
  });
});
