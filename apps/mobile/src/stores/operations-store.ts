import {create} from 'zustand';
import {api} from '../services/api-client';

export type OpType = 'training' | 'inference' | 'download' | 'import' | 'batch' | 'other';
export type OpStatus = 'registered' | 'running' | 'cancelling' | 'cancelled' | 'completed' | 'failed';

export interface Operation {
  id: string;
  type: OpType;
  label: string;
  status: OpStatus;
  created_at: number;
  started_at: number | null;
  finished_at: number | null;
  elapsed_s: number;
  error: string | null;
  meta: Record<string, unknown>;
}

interface OperationsState {
  operations: Operation[];
  counts: Record<string, number>;
  loading: boolean;
  error: string | null;
  _pollTimer: ReturnType<typeof setInterval> | null;

  fetch: () => Promise<void>;
  cancel: (opId: string) => Promise<boolean>;
  cancelAll: (type?: OpType) => Promise<number>;
  startPolling: (intervalMs?: number) => void;
  stopPolling: () => void;
  activeByType: (type: OpType) => Operation[];
  isAnyActive: (type?: OpType) => boolean;
  hasActive: (type: OpType) => boolean;
}

export const useOperationsStore = create<OperationsState>((set, get) => ({
  operations: [],
  counts: {},
  loading: false,
  error: null,
  _pollTimer: null,

  fetch: async () => {
    try {
      set({loading: true, error: null});
      const res = await api.get<{operations: Operation[]; counts: Record<string, number>}>('/operations');
      set({operations: res.operations, counts: res.counts, loading: false});
    } catch (err) {
      set({
        error: err instanceof Error ? err.message : 'Failed to fetch operations',
        loading: false,
      });
    }
  },

  cancel: async (opId: string) => {
    try {
      await api.post(`/cancel/${opId}`);
      await get().fetch();
      return true;
    } catch {
      return false;
    }
  },

  cancelAll: async (type?: OpType) => {
    try {
      const qs = type ? `?type=${type}` : '';
      const res = await api.post<{cancelled: string[]; count: number}>(`/cancel-all${qs}`);
      await get().fetch();
      return res.count;
    } catch {
      return 0;
    }
  },

  startPolling: (intervalMs = 3000) => {
    const existing = get()._pollTimer;
    if (existing) clearInterval(existing);
    get().fetch();
    const timer = setInterval(() => get().fetch(), intervalMs);
    set({_pollTimer: timer});
  },

  stopPolling: () => {
    const timer = get()._pollTimer;
    if (timer) clearInterval(timer);
    set({_pollTimer: null});
  },

  activeByType: (type: OpType) => {
    return get().operations.filter(
      op => op.type === type && ['registered', 'running', 'cancelling'].includes(op.status),
    );
  },

  isAnyActive: (type?: OpType) => {
    return get().operations.some(
      op =>
        ['registered', 'running', 'cancelling'].includes(op.status) &&
        (!type || op.type === type),
    );
  },

  hasActive: (type: OpType) => {
    return get().operations.some(
      op => op.type === type && ['registered', 'running', 'cancelling'].includes(op.status),
    );
  },
}));

export function useActiveOperations(type?: OpType): Operation[] {
  return useOperationsStore(s =>
    s.operations.filter(
      op =>
        ['registered', 'running', 'cancelling'].includes(op.status) &&
        (!type || op.type === type),
    ),
  );
}

export function useHasActiveOperations(type?: OpType): boolean {
  return useOperationsStore(s =>
    s.operations.some(
      op =>
        ['registered', 'running', 'cancelling'].includes(op.status) &&
        (!type || op.type === type),
    ),
  );
}
