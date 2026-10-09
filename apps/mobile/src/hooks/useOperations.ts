import {useEffect, useCallback} from 'react';
import {
  useOperationsStore,
  useActiveOperations,
  useHasActiveOperations,
  type Operation,
  type OpType,
} from '../stores/operations-store';

export function useOperations(type?: OpType, pollIntervalMs = 3000) {
  const operations = useOperationsStore(s => s.operations);
  const counts = useOperationsStore(s => s.counts);
  const loading = useOperationsStore(s => s.loading);
  const error = useOperationsStore(s => s.error);
  const fetchOps = useOperationsStore(s => s.fetch);
  const cancel = useOperationsStore(s => s.cancel);
  const cancelAll = useOperationsStore(s => s.cancelAll);
  const startPolling = useOperationsStore(s => s.startPolling);
  const stopPolling = useOperationsStore(s => s.stopPolling);

  const activeOps = useActiveOperations(type);
  const isActive = useHasActiveOperations(type);
  const hasTraining = useHasActiveOperations('training');
  const hasInference = useHasActiveOperations('inference');

  useEffect(() => {
    startPolling(pollIntervalMs);
    return () => stopPolling();
  }, [startPolling, stopPolling, pollIntervalMs]);

  const cancelOp = useCallback(
    async (opId: string) => {
      const ok = await cancel(opId);
      await fetchOps();
      return ok;
    },
    [cancel, fetchOps],
  );

  const cancelAllByType = useCallback(
    async (cancelType?: OpType) => {
      const n = await cancelAll(cancelType ?? type);
      await fetchOps();
      return n;
    },
    [cancelAll, fetchOps, type],
  );

  const refresh = useCallback(() => fetchOps(), [fetchOps]);

  return {
    operations,
    activeOps,
    counts,
    loading,
    error,
    isActive,
    hasTraining,
    hasInference,
    cancel: cancelOp,
    cancelAll: cancelAllByType,
    refresh,
  };
}
