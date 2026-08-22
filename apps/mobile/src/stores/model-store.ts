import {create} from 'zustand';
import {api} from '../services/api-client';
import type {ModelInfo, SoulInfo, CheckpointInfo, HealthStatus} from '../types';

interface ModelState {
  models: ModelInfo[];
  currentModel: string | null;
  souls: SoulInfo[];
  currentSoul: SoulInfo | null;
  checkpoints: CheckpointInfo[];
  health: HealthStatus | null;
  loading: boolean;
  loadingModelId: string | null;
  switchingSoul: string | null;
  error: string | null;
  lastRefresh: number;
  _refreshAbort: AbortController | null;

  refresh: () => Promise<void>;
  loadModel: (modelId: string) => Promise<boolean>;
  unloadModel: () => Promise<boolean>;
  switchSoul: (name: string, checkpointName?: string) => Promise<boolean>;
  clearError: () => void;
  getModelById: (id: string) => ModelInfo | undefined;
  isModelLoaded: (id: string) => boolean;
}

export const useModelStore = create<ModelState>((set, get) => ({
  models: [],
  currentModel: null,
  souls: [],
  currentSoul: null,
  checkpoints: [],
  health: null,
  loading: false,
  loadingModelId: null,
  switchingSoul: null,
  error: null,
  lastRefresh: 0,
  _refreshAbort: null,

  refresh: async () => {
    // Cancel any in-flight refresh
    const existing = get()._refreshAbort;
    if (existing) existing.abort();
    const abort = new AbortController();
    set({_refreshAbort: abort});

    set({loading: true, error: null});
    try {
      const [models, soulsData, currentSoul, checkpoints, health] =
        await Promise.all([
          api.get<ModelInfo[]>('/models').catch(() => []),
          api.get<{souls: SoulInfo[]; current_soul: string | null}>('/souls').catch(() => ({souls: [], current_soul: null})),
          api.get<SoulInfo>('/souls/current').catch(() => null),
          api.get<CheckpointInfo[]>('/auto-train/checkpoints').catch(() => []),
          api.get<HealthStatus>('/health').catch(() => null),
        ]);

      // Check if aborted before setting state
      if (abort.signal.aborted) return;

      set({
        models,
        souls: Array.isArray(soulsData) ? soulsData : (soulsData.souls || []),
        currentSoul,
        checkpoints,
        health,
        currentModel: health?.model_type || null,
        loading: false,
        loadingModelId: null,
        switchingSoul: null,
        lastRefresh: Date.now(),
      });
    } catch (err: any) {
      if (err.name === 'AbortError') return;
      set({error: err.message, loading: false, loadingModelId: null, switchingSoul: null});
    }
  },

  loadModel: async (modelId: string) => {
    // Optimistic: set loading immediately
    set({loadingModelId: modelId, error: null});
    try {
      await api.post('/models/load', {model_id: modelId});
      // Optimistic: assume load succeeded, update currentModel immediately
      set({currentModel: modelId});
      await get().refresh();
      return true;
    } catch (err: any) {
      // Rollback optimistic update on failure
      set({error: err.message, loadingModelId: null, currentModel: get().health?.model_type || null});
      return false;
    }
  },

  unloadModel: async () => {
    set({error: null});
    // Optimistic: clear current model immediately
    const prevModel = get().currentModel;
    set({currentModel: null});
    try {
      await api.post('/models/unload');
      await get().refresh();
      return true;
    } catch (err: any) {
      // Rollback on failure
      set({error: err.message, currentModel: prevModel});
      return false;
    }
  },

  switchSoul: async (name: string, checkpointName?: string) => {
    set({switchingSoul: name, error: null});
    // Optimistic: update currentSoul immediately
    const prevSoul = get().currentSoul;
    const optimisticSoul = get().souls.find(s => s.name === name) || null;
    if (optimisticSoul) {
      set({currentSoul: optimisticSoul});
    }
    try {
      await api.post('/souls/switch', {
        soul: name,
        checkpoint_name: checkpointName || null,
      });
      await get().refresh();
      return true;
    } catch (err: any) {
      // Rollback on failure
      set({error: err.message, switchingSoul: null, currentSoul: prevSoul});
      return false;
    }
  },

  clearError: () => set({error: null}),

  getModelById: (id: string) => {
    return get().models.find(m => m.id === id || m.name === id);
  },

  isModelLoaded: (id: string) => {
    const current = get().currentModel;
    if (!current) return false;
    return current === id || current.includes(id);
  },
}));
