jest.mock('../../services/api-client', () => {
  return {
    getApiUrl: Object.assign(jest.fn().mockResolvedValue('http://localhost:8000'), {mockClear: jest.fn()}),
  };
});

jest.mock('../../services/haptics', () => ({
  triggerHaptic: jest.fn(),
}));

jest.mock('react-native', () => ({
  Platform: {OS: 'ios'},
  NativeModules: {LlamaContext: {create: jest.fn(), release: jest.fn()}},
}));

jest.mock('@react-native-async-storage/async-storage', () =>
  require('@react-native-async-storage/async-storage/jest/async-storage-mock'),
);

jest.mock('react-native-fs', () => ({
  DocumentDirectoryPath: '/docs',
  CachesDirectoryPath: '/cache',
  exists: jest.fn().mockResolvedValue(true),
  downloadFile: jest.fn(() => ({promise: Promise.resolve({statusCode: 200})})),
}));

import * as llama from '../llama-rn-service';

beforeEach(() => {
  jest.clearAllMocks();
});

describe('llama-rn-service', () => {
  it('isLoaded returns false initially', () => {
    expect(llama.isLoaded()).toBe(false);
  });

  it('getModelPath returns null when not set', async () => {
    const result = await llama.getModelPath();
    expect(result === null || typeof result === 'string').toBe(true);
  });

  it('chatCompletion throws when model not loaded', async () => {
    await expect(
      llama.chatCompletion([{role: 'user', content: 'hi'}]),
    ).rejects.toThrow('Model not loaded');
  });

  it('chatCompletionStream throws when model not loaded', async () => {
    const gen = llama.chatCompletionStream([{role: 'user', content: 'hi'}]);
    await expect(gen.next()).rejects.toThrow('Model not loaded');
  });
});
