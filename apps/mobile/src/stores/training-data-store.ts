/**
 * Training data store — structured Zustand store for on-device training pairs.
 *
 * Replaces raw AsyncStorage array with indexed, queryable storage.
 * Persists via AsyncStorage but with proper structure and batch ops.
 *
 * Schema per pair:
 *   { id, user_msg, assistant_msg, quality, timestamp, session_id, synced }
 *
 * Indexed access:
 *   - by ID: O(1) lookup
 *   - by session: O(n) scan but cached
 *   - by quality: filtered views
 *   - pending sync: unsent pairs only
 */

import {create} from 'zustand';
import AsyncStorage from '@react-native-async-storage/async-storage';

// ── Types ─────────────────────────────────────────────────────────────

export interface TrainingPair {
  id: string;
  user_msg: string;
  assistant_msg: string;
  quality: number;  // 1 = thumbs up, -1 = thumbs down, 0 = neutral
  timestamp: number;
  session_id: string;
  /** Whether this pair has been sent to server for training */
  synced: boolean;
}

export interface TrainingDataState {
  /** All pairs indexed by ID */
  pairs: Record<string, TrainingPair>;
  /** Ordered list of IDs (newest first) */
  pairIds: string[];
  /** Whether initial load from storage is complete */
  hydrated: boolean;
  /** Last training trigger result */
  lastTrainResult: {loss: number; checkpoint: string; timestamp: number} | null;

  // ── Actions ───────────────────────────────────────────────────────
  addPair: (userMsg: string, assistantMsg: string, sessionId: string, quality?: number) => string;
  updateQuality: (pairId: string, quality: number) => void;
  markSynced: (pairIds: string[]) => void;
  removePair: (pairId: string) => void;
  clearSynced: () => void;
  clearAll: () => void;
  setLastTrainResult: (result: {loss: number; checkpoint: string; timestamp: number}) => void;

  // ── Selectors ─────────────────────────────────────────────────────
  getPendingPairs: () => TrainingPair[];
  getPairsBySession: (sessionId: string) => TrainingPair[];
  getPairsByQuality: (minQuality: number) => TrainingPair[];
  getStats: () => {total: number; pending: number; synced: number; byQuality: Record<number, number>};
}

// ── Storage ───────────────────────────────────────────────────────────

const STORAGE_KEY = '@sloughgpt/training_data';

let _idCounter = 0;
function _nextId(): string {
  _idCounter++;
  return `tp_${Date.now()}_${_idCounter}`;
}

// ── Persistence helpers ───────────────────────────────────────────────

function _toPersist(state: TrainingDataState): string {
  return JSON.stringify({
    pairs: state.pairs,
    pairIds: state.pairIds,
    lastTrainResult: state.lastTrainResult,
  });
}

let _persistTimer: ReturnType<typeof setTimeout> | null = null;

function _debouncedPersist(state: TrainingDataState) {
  if (_persistTimer) clearTimeout(_persistTimer);
  _persistTimer = setTimeout(() => {
    AsyncStorage.setItem(STORAGE_KEY, _toPersist(state));
  }, 200);
}

// ── Store ─────────────────────────────────────────────────────────────

export const useTrainingDataStore = create<TrainingDataState>((set, get) => ({
  pairs: {},
  pairIds: [],
  hydrated: false,
  lastTrainResult: null,

  addPair: (userMsg, assistantMsg, sessionId, quality = 0) => {
    const id = _nextId();
    const pair: TrainingPair = {
      id,
      user_msg: userMsg,
      assistant_msg: assistantMsg,
      quality,
      timestamp: Date.now(),
      session_id: sessionId,
      synced: false,
    };

    set(state => {
      const next = {
        pairs: {...state.pairs, [id]: pair},
        pairIds: [id, ...state.pairIds],
      };
      _debouncedPersist({...state, ...next});
      return next;
    });

    return id;
  },

  updateQuality: (pairId, quality) => {
    set(state => {
      const pair = state.pairs[pairId];
      if (!pair) return state;
      const next = {
        pairs: {...state.pairs, [pairId]: {...pair, quality}},
      };
      _debouncedPersist({...state, ...next});
      return next;
    });
  },

  markSynced: (pairIds) => {
    set(state => {
      const nextPairs = {...state.pairs};
      for (const id of pairIds) {
        if (nextPairs[id]) nextPairs[id] = {...nextPairs[id], synced: true};
      }
      const next = {pairs: nextPairs};
      _debouncedPersist({...state, ...next});
      return next;
    });
  },

  removePair: (pairId) => {
    set(state => {
      const {[pairId]: _, ...rest} = state.pairs;
      const next = {
        pairs: rest,
        pairIds: state.pairIds.filter(id => id !== pairId),
      };
      _debouncedPersist({...state, ...next});
      return next;
    });
  },

  clearSynced: () => {
    set(state => {
      const remaining: Record<string, TrainingPair> = {};
      const remainingIds: string[] = [];
      for (const id of state.pairIds) {
        if (!state.pairs[id]?.synced) {
          remaining[id] = state.pairs[id];
          remainingIds.push(id);
        }
      }
      const next = {pairs: remaining, pairIds: remainingIds};
      _debouncedPersist({...state, ...next});
      return next;
    });
  },

  clearAll: () => {
    const next = {pairs: {}, pairIds: [], lastTrainResult: null};
    set(next);
    AsyncStorage.removeItem(STORAGE_KEY);
  },

  setLastTrainResult: (result) => {
    set(state => {
      const next = {lastTrainResult: result};
      _debouncedPersist({...state, ...next});
      return next;
    });
  },

  // ── Selectors ─────────────────────────────────────────────────────

  getPendingPairs: () => {
    const {pairs, pairIds} = get();
    return pairIds
      .map(id => pairs[id])
      .filter(p => p && !p.synced);
  },

  getPairsBySession: (sessionId) => {
    const {pairs, pairIds} = get();
    return pairIds
      .map(id => pairs[id])
      .filter(p => p?.session_id === sessionId);
  },

  getPairsByQuality: (minQuality) => {
    const {pairs, pairIds} = get();
    return pairIds
      .map(id => pairs[id])
      .filter(p => p && p.quality >= minQuality);
  },

  getStats: () => {
    const {pairs, pairIds} = get();
    let synced = 0;
    const byQuality: Record<number, number> = {};
    for (const id of pairIds) {
      const p = pairs[id];
      if (!p) continue;
      if (p.synced) synced++;
      byQuality[p.quality] = (byQuality[p.quality] || 0) + 1;
    }
    return {
      total: pairIds.length,
      pending: pairIds.length - synced,
      synced,
      byQuality,
    };
  },
}));

// ── Hydration ─────────────────────────────────────────────────────────

let _hydrated = false;

export async function hydrateTrainingData(): Promise<void> {
  if (_hydrated) return;
  _hydrated = true;

  try {
    const raw = await AsyncStorage.getItem(STORAGE_KEY);
    if (raw) {
      const data = JSON.parse(raw);
      useTrainingDataStore.setState({
        pairs: data.pairs || {},
        pairIds: data.pairIds || [],
        lastTrainResult: data.lastTrainResult || null,
        hydrated: true,
      });
    } else {
      useTrainingDataStore.setState({hydrated: true});
    }
  } catch {
    useTrainingDataStore.setState({hydrated: true});
  }
}
