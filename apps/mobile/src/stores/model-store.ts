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
  error: string | null;
  refresh: () => Promise<void>;
  loadModel: (modelId: string) => Promise<void>;
  unloadModel: () => Promise<void>;
  switchSoul: (name: string, checkpointName?: string) => Promise<void>;
  clearError: () => void;
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
  error: null,

  refresh: async () => {
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
      set({
        models,
        souls: Array.isArray(soulsData) ? soulsData : (soulsData.souls || []),
        currentSoul,
        checkpoints,
        health,
        currentModel: health?.model_type || null,
        loading: false,
      });
    } catch (err: any) {
      set({error: err.message, loading: false});
    }
  },

  loadModel: async (modelId: string) => {
    set({loadingModelId: modelId, error: null});
    try {
      await api.post('/models/load', {model_id: modelId});
      await get().refresh();
    } catch (err: any) {
      set({error: err.message, loadingModelId: null});
    }
  },

  unloadModel: async () => {
    set({error: null});
    try {
      await api.post('/models/unload');
      await get().refresh();
    } catch (err: any) {
      set({error: err.message});
    }
  },

  switchSoul: async (name: string, checkpointName?: string) => {
    set({error: null});
    try {
      await api.post('/souls/switch', {
        soul: name,
        checkpoint_name: checkpointName || null,
      });
      await get().refresh();
    } catch (err: any) {
      set({error: err.message});
    }
  },

  clearError: () => set({error: null}),
}));
