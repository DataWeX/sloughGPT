/**
 * Tests for the hybrid inference store.
 *
 * Uses zustand store directly (no React render — stores are testable pure JS).
 */

import AsyncStorage from '@react-native-async-storage/async-storage';
import * as slonet from '../../services/onnx-inference-service';
import * as llamaRn from '../../services/llama-rn-service';
import * as souLoader from '../../services/sou-loader';

jest.mock('../../services/onnx-inference-service');
jest.mock('../../services/llama-rn-service');
jest.mock('../../services/sou-loader');

const mockSloNet = slonet as jest.Mocked<typeof slonet>;
const mockLlamaRn = llamaRn as jest.Mocked<typeof llamaRn>;
const mockSouLoader = souLoader as jest.Mocked<typeof souLoader>;

// eslint-disable-next-line @typescript-eslint/no-var-requires
const {useHybridStore} = require('../hybrid-inference-store');

const INITIAL = {
  slonet: {kind: 'slonet', loaded: false, modelName: '', downloadProgress: null, description: expect.any(String)},
  qwen: {kind: 'qwen', loaded: false, modelName: 'Qwen2.5-0.5B-Instruct', downloadProgress: null, description: expect.any(String)},
  activeEngine: 'remote',
  downloadProgress: 0,
  lastError: null,
};

beforeEach(() => {
  jest.clearAllMocks();
  (AsyncStorage.getItem as jest.Mock).mockResolvedValue(null);
  (AsyncStorage.setItem as jest.Mock).mockResolvedValue(undefined);
  // Reset store to initial state
  useHybridStore.setState({
    slonet: {kind: 'slonet', loaded: false, modelName: '', downloadProgress: null, description: 'Baby transformer (fast, local, any prompt)'},
    qwen: {kind: 'qwen', loaded: false, modelName: 'Qwen2.5-0.5B-Instruct', downloadProgress: null, description: '500M chat model (local, any prompt)'},
    activeEngine: 'remote',
    downloadProgress: 0,
    lastError: null,
  });
});

// ── Initial state ────────────────────────────────────────────────────────

describe('initial state', () => {
  it('starts with remote active and no engines loaded', () => {
    const state = useHybridStore.getState();
    expect(state.activeEngine).toBe('remote');
    expect(state.slonet.loaded).toBe(false);
    expect(state.qwen.loaded).toBe(false);
  });

  it('has valid descriptions', () => {
    const state = useHybridStore.getState();
    expect(state.slonet.description).toContain('any prompt');
    expect(state.qwen.description).toContain('any prompt');
  });
});

// ── setActiveEngine ──────────────────────────────────────────────────────

describe('setActiveEngine', () => {
  it('updates state and persists to AsyncStorage', async () => {
    await useHybridStore.getState().setActiveEngine('qwen');
    expect(useHybridStore.getState().activeEngine).toBe('qwen');
    expect(AsyncStorage.setItem).toHaveBeenCalledWith(
      '@sloughgpt/hybrid_config',
      JSON.stringify({activeEngine: 'qwen', offlineOnly: false}),
    );
  });
});

// ── decideRoute ──────────────────────────────────────────────────────────

describe('decideRoute', () => {
  it('returns remote when activeEngine is remote', () => {
    const route = useHybridStore.getState().decideRoute('any content');
    expect(route).toEqual({target: 'remote', reason: 'user selected remote'});
  });

  it('returns slonet when active and loaded', async () => {
    mockSloNet.loadCheckpoint.mockResolvedValue();
    await useHybridStore.getState().loadSloNet('test-cp');
    await useHybridStore.getState().setActiveEngine('slonet');

    const route = useHybridStore.getState().decideRoute('any prompt whatsoever');
    expect(route).toEqual({target: 'local', engine: 'slonet'});
  });

  it('returns qwen when active and loaded', async () => {
    mockLlamaRn.downloadModel.mockResolvedValue();
    mockLlamaRn.loadModel.mockResolvedValue();
    await useHybridStore.getState().loadQwen();
    await useHybridStore.getState().setActiveEngine('qwen');

    const route = useHybridStore.getState().decideRoute('any prompt');
    expect(route).toEqual({target: 'local', engine: 'qwen'});
  });

  it('falls back to slonet when qwen selected but not loaded', async () => {
    mockSloNet.loadCheckpoint.mockResolvedValue();
    await useHybridStore.getState().loadSloNet();
    await useHybridStore.getState().setActiveEngine('qwen');

    const route = useHybridStore.getState().decideRoute('any prompt');
    expect(route).toEqual({target: 'local', engine: 'slonet'});
  });

  it('falls back to remote when selected engine not loaded and no alternatives', async () => {
    await useHybridStore.getState().setActiveEngine('slonet');
    const route = useHybridStore.getState().decideRoute('any');
    expect(route).toEqual({target: 'remote', reason: 'no local engine loaded'});
  });
});

// ── loadSloNet ───────────────────────────────────────────────────────────

describe('loadSloNet', () => {
  it('marks slonet as loaded on success', async () => {
    mockSloNet.loadCheckpoint.mockResolvedValue();
    await useHybridStore.getState().loadSloNet('baby-slonet');

    const s = useHybridStore.getState();
    expect(s.slonet.loaded).toBe(true);
    expect(s.slonet.modelName).toBe('baby-slonet');
    expect(s.lastError).toBeNull();
  });

  it('sets error on failure', async () => {
    mockSloNet.loadCheckpoint.mockRejectedValue(new Error('bad checkpoint'));
    await useHybridStore.getState().loadSloNet('broken');

    const s = useHybridStore.getState();
    expect(s.slonet.loaded).toBe(false);
    expect(s.lastError).toBe('bad checkpoint');
  });
});

// ── loadSloNetFromSou ────────────────────────────────────────────────────

describe('loadSloNetFromSou', () => {
  it('marks slonet as loaded when picker succeeds', async () => {
    mockSouLoader.pickAndLoadSou.mockResolvedValue({ name: 'my-model', config: { n_embed: 128, n_head: 4, n_layer: 2, vocab_size: 100, block_size: 64 } });
    await useHybridStore.getState().loadSloNetFromSou();

    const s = useHybridStore.getState();
    expect(s.slonet.loaded).toBe(true);
    expect(s.slonet.modelName).toBe('my-model');
  });

  it('sets error when picker throws', async () => {
    mockSouLoader.pickAndLoadSou.mockRejectedValue(new Error('picker failed'));
    await useHybridStore.getState().loadSloNetFromSou();

    const s = useHybridStore.getState();
    expect(s.slonet.loaded).toBe(false);
    expect(s.lastError).toBe('picker failed');
  });
});

// ── loadQwen ─────────────────────────────────────────────────────────────

describe('loadQwen', () => {
  it('marks qwen as loaded on success', async () => {
    mockLlamaRn.downloadModel.mockResolvedValue();
    mockLlamaRn.loadModel.mockResolvedValue();
    await useHybridStore.getState().loadQwen();

    const s = useHybridStore.getState();
    expect(s.qwen.loaded).toBe(true);
    expect(s.lastError).toBeNull();
  });

  it('sets error on download failure', async () => {
    mockLlamaRn.downloadModel.mockRejectedValue(new Error('no network'));
    await useHybridStore.getState().loadQwen();

    const s = useHybridStore.getState();
    expect(s.qwen.loaded).toBe(false);
    expect(s.lastError).toContain('no network');
  });
});

// ── unload ───────────────────────────────────────────────────────────────

describe('unload', () => {
  beforeEach(async () => {
    mockSloNet.loadCheckpoint.mockResolvedValue();
    mockLlamaRn.downloadModel.mockResolvedValue();
    mockLlamaRn.loadModel.mockResolvedValue();
    await useHybridStore.getState().loadSloNet();
    await useHybridStore.getState().loadQwen();
  });

  it('unloadSloNet unloads and updates state', () => {
    useHybridStore.getState().unloadSloNet();
    const s = useHybridStore.getState();
    expect(s.slonet.loaded).toBe(false);
    expect(mockSloNet.unload).toHaveBeenCalled();
  });

  it('unloadQwen unloads and updates state', async () => {
    await useHybridStore.getState().unloadQwen();
    const s = useHybridStore.getState();
    expect(s.qwen.loaded).toBe(false);
    expect(mockLlamaRn.unloadModel).toHaveBeenCalled();
  });

  it('unloadAll unloads both engines', async () => {
    await useHybridStore.getState().unloadAll();
    const s = useHybridStore.getState();
    expect(s.slonet.loaded).toBe(false);
    expect(s.qwen.loaded).toBe(false);
    expect(mockSloNet.unload).toHaveBeenCalled();
    expect(mockLlamaRn.unloadModel).toHaveBeenCalled();
  });
});

// ── executeLocal ─────────────────────────────────────────────────────────

describe('executeLocal', () => {
  it('returns null when route is remote', async () => {
    const res = await useHybridStore.getState().executeLocal('hi', []);
    expect(res).toBeNull();
  });

  it('delegates to slonet', async () => {
    mockSloNet.loadCheckpoint.mockResolvedValue();
    mockSloNet.generate.mockResolvedValue({
      text: 'hello back',
      tokens_generated: 3,
      elapsed_ms: 15,
    });

    await useHybridStore.getState().loadSloNet();
    await useHybridStore.getState().setActiveEngine('slonet');

    const res = await useHybridStore.getState().executeLocal('hi', []);
    expect(res).toEqual({text: 'hello back', tokens_generated: 3, elapsed_ms: 15});
    expect(mockSloNet.generate).toHaveBeenCalledWith('hi', 64, 0.8, 40, 0.9, 0, undefined);
  });

  it('delegates to qwen', async () => {
    mockLlamaRn.downloadModel.mockResolvedValue();
    mockLlamaRn.loadModel.mockResolvedValue();
    mockLlamaRn.chatCompletion.mockResolvedValue({
      text: 'I am Qwen',
      tokensGenerated: 5,
      elapsedMs: 200,
    });

    await useHybridStore.getState().loadQwen();
    await useHybridStore.getState().setActiveEngine('qwen');

    const res = await useHybridStore.getState().executeLocal('hello', [
      {role: 'user', content: 'hello'},
    ]);
    expect(res).toEqual({text: 'I am Qwen', tokens_generated: 5, elapsed_ms: 200});
    expect(mockLlamaRn.chatCompletion).toHaveBeenCalled();
  });

  it('sets lastError on failure', async () => {
    mockSloNet.loadCheckpoint.mockResolvedValue();
    mockSloNet.generate.mockRejectedValue(new Error('OOM'));

    await useHybridStore.getState().loadSloNet();
    await useHybridStore.getState().setActiveEngine('slonet');

    const res = await useHybridStore.getState().executeLocal('hi', []);
    expect(res).toBeNull();
    expect(useHybridStore.getState().lastError).toBe('OOM');
  });
});
