/**
 * Hybrid inference store — manages which inference engine handles each request.
 *
 * No routing heuristics — user picks the active engine (SloNet / Qwen / Remote).
 * SloNet handles any prompt (~5ms/token, lower quality).
 * Qwen handles any prompt (~15-30 tok/s via Metal, higher quality).
 * Remote handles any prompt via server.
 *
 * Performance: O(1) — just returns the active engine if loaded, no string analysis.
 */

import {create} from 'zustand';
import AsyncStorage from '@react-native-async-storage/async-storage';
import * as slonet from '../services/onnx-inference-service';
import * as llamaRn from '../services/llama-rn-service';
import type {LocalGenerateResult, HybridState, RoutingDecision, ActiveEngine, OnTokenCallback} from '../types/local-inference';

const STORAGE_KEY = '@sloughgpt/hybrid_config';

// ── Pure routing (O(1), no string scanning at all) ─────────────────────

function _route(_content: string, state: HybridState): RoutingDecision {
  // Offline-only mode: never route to remote
  if (state.offlineOnly) {
    if (state.slonet.loaded) return {target: 'local', engine: 'slonet'};
    if (state.qwen.loaded) return {target: 'local', engine: 'qwen'};
    return {target: 'remote', reason: 'offline-only — load a local engine in Settings'};
  }

  if (state.activeEngine === 'remote') {
    return {target: 'remote', reason: 'user selected remote'};
  }

  if (state.activeEngine === 'slonet' && state.slonet.loaded) {
    return {target: 'local', engine: 'slonet'};
  }

  if (state.activeEngine === 'qwen' && state.qwen.loaded) {
    return {target: 'local', engine: 'qwen'};
  }

  // Selected engine not loaded — try the other local engine, then remote
  if (state.slonet.loaded) return {target: 'local', engine: 'slonet'};
  if (state.qwen.loaded) return {target: 'local', engine: 'qwen'};
  return {target: 'remote', reason: 'no local engine loaded'};
}

// ── Store ───────────────────────────────────────────────────────────────

interface HybridStoreState extends HybridState {
  downloadProgress: number;
  lastError: string | null;

  setActiveEngine: (engine: ActiveEngine) => Promise<void>;
  setOfflineOnly: (enabled: boolean) => Promise<void>;
  loadSloNet: (checkpointName?: string) => Promise<void>;
  loadQwen: (onProgress?: (f: number) => void) => Promise<void>;
  unloadSloNet: () => void;
  unloadQwen: () => Promise<void>;
  unloadAll: () => Promise<void>;
  decideRoute: (content: string) => RoutingDecision;
  executeLocal: (
    content: string,
    messages: Array<{role: string; content: string}>,
    onToken?: OnTokenCallback,
  ) => Promise<LocalGenerateResult | null>;
}

export const useHybridStore = create<HybridStoreState>((set, get) => ({
  slonet: {
    kind: 'slonet',
    loaded: false,
    modelName: '',
    downloadProgress: null,
    description: 'Baby transformer (fast, local, any prompt)',
  },
  qwen: {
    kind: 'qwen',
    loaded: false,
    modelName: 'Qwen2.5-0.5B-Instruct',
    downloadProgress: null,
    description: '500M chat model (local, any prompt)',
  },
  activeEngine: 'remote',
  offlineOnly: false,
  downloadProgress: 0,
  lastError: null,

  setActiveEngine: async engine => {
    if (get().offlineOnly && engine === 'remote') return;
    set({activeEngine: engine});
    await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify({activeEngine: engine, offlineOnly: get().offlineOnly}));
  },

  setOfflineOnly: async enabled => {
    set({offlineOnly: enabled});
    if (enabled && get().activeEngine === 'remote') {
      set({activeEngine: 'slonet'});
    }
    await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify({activeEngine: get().activeEngine, offlineOnly: enabled}));
  },

  loadSloNet: async (checkpointName?: string) => {
    set({downloadProgress: 0, lastError: null});
    try {
      await slonet.loadCheckpoint(checkpointName || 'baby-slonet');
      set({
        slonet: {...get().slonet, loaded: true, modelName: checkpointName || 'baby-slonet'},
        downloadProgress: 1,
      });
    } catch (err: any) {
      set({lastError: err.message, downloadProgress: 0});
    }
  },

  loadQwen: async onProgress => {
    set({downloadProgress: 0, lastError: null});
    try {
      const cb = onProgress || ((f: number) => set({downloadProgress: f}));
      await llamaRn.downloadModel(cb);
      await llamaRn.loadModel();
      set({
        qwen: {...get().qwen, loaded: true},
        downloadProgress: 1,
      });
    } catch (err: any) {
      set({lastError: `Qwen load failed: ${err.message}`, downloadProgress: 0});
    }
  },

  unloadSloNet: () => {
    slonet.unload();
    set({slonet: {...get().slonet, loaded: false}});
  },

  unloadQwen: async () => {
    await llamaRn.unloadModel();
    set({qwen: {...get().qwen, loaded: false}});
  },

  unloadAll: async () => {
    slonet.unload();
    await llamaRn.unloadModel();
    set({slonet: {...get().slonet, loaded: false}, qwen: {...get().qwen, loaded: false}});
  },

  decideRoute: (content: string) => _route(content, get()),

  executeLocal: async (content, messages, onToken) => {
    const route = get().decideRoute(content);
    if (route.target !== 'local') return null;

    try {
      if (route.engine === 'slonet') {
        const result = await slonet.generate(content, 64, 0.8, 40, 0.9, 0, onToken);
        return result;
      }

      if (route.engine === 'qwen') {
        const result = await llamaRn.chatCompletion(messages, undefined, onToken);
        return {
          text: result.text,
          tokens_generated: result.tokensGenerated,
          elapsed_ms: result.elapsedMs,
        };
      }

      return null;
    } catch (err: any) {
      set({lastError: err.message});
      return null;
    }
  },
}));

// ── Persist active engine ───────────────────────────────────────────────

(async function hydrate() {
  try {
    const raw = await AsyncStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (parsed.offlineOnly) {
        useHybridStore.setState({offlineOnly: true, activeEngine: 'slonet'});
      } else if (parsed.activeEngine) {
        useHybridStore.getState().setActiveEngine(parsed.activeEngine);
      }
    }
  } catch {}
})();
