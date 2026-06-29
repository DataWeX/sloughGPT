import AsyncStorage from '@react-native-async-storage/async-storage';

// Mock api-client
jest.mock('../../services/api-client', () => ({
  getApiUrl: jest.fn(async () => 'http://localhost:8000'),
}));

// Mock react-native-sensors to throw so it falls back to mock
jest.mock('react-native-sensors', () => {
  throw new Error('not installed');
});

// Mock global fetch
const mockFetch = jest.fn();
global.fetch = mockFetch as any;

beforeEach(async () => {
  jest.clearAllMocks();
  (AsyncStorage as any)._store = {};
  mockFetch.mockResolvedValue({ok: true, json: async () => ({})});
});

const mod = require('../background-recorder');

afterEach(async () => {
  await mod.stopBackgroundRecording();
});

describe('background-recorder', () => {
  describe('getBackgroundRecorderState', () => {
    it('returns inactive state initially', () => {
      const state = mod.getBackgroundRecorderState();
      expect(state.active).toBe(false);
      expect(state.bufferSize).toBe(0);
      expect(state.lastSync).toBeNull();
    });
  });

  describe('startBackgroundRecording', () => {
    it('marks recording as active in AsyncStorage', async () => {
      const cleanup = await mod.startBackgroundRecording();

      const flag = await AsyncStorage.getItem('@sloughgpt/recording_active');
      expect(flag).toBe('true');

      cleanup();
    });

    it('returns noop cleanup if already recording', async () => {
      await AsyncStorage.setItem('@sloughgpt/recording_active', 'true');

      const cleanup = await mod.startBackgroundRecording();
      expect(typeof cleanup).toBe('function');
    });
  });

  describe('stopBackgroundRecording', () => {
    it('sets recording flag to false', async () => {
      await mod.startBackgroundRecording();
      await mod.stopBackgroundRecording();

      const flag = await AsyncStorage.getItem('@sloughgpt/recording_active');
      expect(flag).toBe('false');
    });
  });

  describe('getBufferSize', () => {
    it('returns 0 when no readings', async () => {
      const size = await mod.getBufferSize();
      expect(size).toBe(0);
    });
  });

  describe('getLastSyncTime', () => {
    it('returns null when no sync has happened', async () => {
      const time = await mod.getLastSyncTime();
      expect(time).toBeNull();
    });

    it('returns stored timestamp', async () => {
      const now = Date.now();
      await AsyncStorage.setItem('@sloughgpt/last_sensor_sync', String(now));

      const time = await mod.getLastSyncTime();
      expect(time).toBe(now);
    });
  });
});
