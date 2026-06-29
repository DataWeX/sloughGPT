import AsyncStorage from '@react-native-async-storage/async-storage';

// Mock api-client
jest.mock('../../services/api-client', () => ({
  getApiUrl: jest.fn(async () => 'http://localhost:8000'),
}));

// Mock global fetch
const mockFetch = jest.fn();
global.fetch = mockFetch as any;

beforeEach(() => {
  jest.clearAllMocks();
  (AsyncStorage as any)._store = {};
  mockFetch.mockResolvedValue({
    ok: true,
    json: async () => ({model_loaded: true, num_recordings: 20}),
  });
});

const mod = require('../auto-train-scheduler');

afterEach(() => {
  mod.stopAutoTrainScheduler();
});

describe('auto-train-scheduler', () => {
  describe('getAutoTrainConfig', () => {
    it('returns defaults when no stored config', async () => {
      const config = await mod.getAutoTrainConfig();
      expect(config.enabled).toBe(true);
      expect(config.intervalMs).toBe(300_000);
      expect(config.minNewRecordings).toBe(10);
      expect(config.cooldownMs).toBe(120_000);
    });

    it('merges stored config with defaults', async () => {
      await AsyncStorage.setItem(
        '@sloughgpt/auto_train_config',
        JSON.stringify({enabled: false, minNewRecordings: 5}),
      );

      const config = await mod.getAutoTrainConfig();
      expect(config.enabled).toBe(false);
      expect(config.minNewRecordings).toBe(5);
      expect(config.intervalMs).toBe(300_000);
    });
  });

  describe('setAutoTrainConfig', () => {
    it('persists partial config', async () => {
      await mod.getAutoTrainConfig();
      await mod.setAutoTrainConfig({enabled: false});

      const raw = await AsyncStorage.getItem('@sloughgpt/auto_train_config');
      const parsed = JSON.parse(raw!);
      expect(parsed.enabled).toBe(false);
    });
  });

  describe('startAutoTrainScheduler / stopAutoTrainScheduler', () => {
    it('starts and stops', () => {
      expect(mod.isAutoTrainRunning()).toBe(false);

      mod.startAutoTrainScheduler();
      expect(mod.isAutoTrainRunning()).toBe(true);

      mod.stopAutoTrainScheduler();
      expect(mod.isAutoTrainRunning()).toBe(false);
    });

    it('is idempotent to start twice', () => {
      mod.startAutoTrainScheduler();
      mod.startAutoTrainScheduler();
      expect(mod.isAutoTrainRunning()).toBe(true);
    });

    it('is idempotent to stop twice', () => {
      mod.startAutoTrainScheduler();
      mod.stopAutoTrainScheduler();
      mod.stopAutoTrainScheduler();
      expect(mod.isAutoTrainRunning()).toBe(false);
    });
  });

  describe('isAutoTrainRunning', () => {
    it('returns false initially', () => {
      expect(mod.isAutoTrainRunning()).toBe(false);
    });

    it('returns true after start', () => {
      mod.startAutoTrainScheduler();
      expect(mod.isAutoTrainRunning()).toBe(true);
    });
  });
});
