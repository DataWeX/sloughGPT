jest.mock('../../services/api-client', () => ({
  getApiUrl: jest.fn().mockResolvedValue('http://localhost:8000'),
}));

jest.mock('@react-native-async-storage/async-storage', () =>
  require('@react-native-async-storage/async-storage/jest/async-storage-mock'),
);

import * as onnx from '../onnx-inference-service';

beforeEach(() => {
  onnx.unload();
});

describe('onnx-inference-service', () => {
  it('isLoaded returns false initially', () => {
    expect(onnx.isLoaded()).toBe(false);
  });

  it('unload clears state', () => {
    onnx.unload();
    expect(onnx.isLoaded()).toBe(false);
  });

  it('generate throws when no checkpoint loaded', async () => {
    await expect(onnx.generate('hello')).rejects.toThrow('No checkpoint loaded');
  });

  it('loadCheckpoint fetches from server', async () => {
    const fetchSpy = jest.spyOn(global, 'fetch').mockResolvedValue({
      ok: true,
      json: async () => ({
        config: {
          vocab_size: 256,
          n_embed: 128,
          n_layer: 1,
          n_head: 4,
          block_size: 128,
        },
        weights_b64: btoa(new Float32Array(256 * 128 + 128 * 128 * 7 + 128 + 128 * 256).buffer),
      }),
    } as Response);
    await onnx.loadCheckpoint('test-ckpt');
    expect(onnx.isLoaded()).toBe(true);
    fetchSpy.mockRestore();
  });
});
