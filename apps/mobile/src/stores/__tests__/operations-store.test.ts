import {
  useOperationsStore,
  type Operation,
} from '../operations-store';
import {api} from '../../services/api-client';

jest.mock('../../services/api-client');

const mockApi = api as jest.Mocked<typeof api>;

function makeOp(overrides: Partial<Operation> = {}): Operation {
  return {
    id: `op_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
    type: 'training',
    label: 'Test Op',
    status: 'registered',
    created_at: Date.now(),
    started_at: null,
    finished_at: null,
    elapsed_s: 0,
    error: null,
    meta: {},
    ...overrides,
  };
}

beforeEach(() => {
  jest.clearAllMocks();
  useOperationsStore.setState({
    operations: [],
    counts: {},
    loading: false,
    error: null,
    _pollTimer: null,
  });
});

describe('operations-store', () => {
  describe('fetch', () => {
    it('fetches operations and updates state', async () => {
      const ops = [makeOp({id: 'op1'}), makeOp({id: 'op2', type: 'inference'})];
      mockApi.get.mockResolvedValueOnce({operations: ops, counts: {training: 1, inference: 1}});

      await useOperationsStore.getState().fetch();

      const state = useOperationsStore.getState();
      expect(state.operations).toHaveLength(2);
      expect(state.counts).toEqual({training: 1, inference: 1});
      expect(state.loading).toBe(false);
      expect(state.error).toBeNull();
    });

    it('handles fetch error', async () => {
      mockApi.get.mockRejectedValueOnce(new Error('network fail'));

      await useOperationsStore.getState().fetch();

      const state = useOperationsStore.getState();
      expect(state.error).toBe('network fail');
      expect(state.loading).toBe(false);
    });

    it('handles non-Error exceptions', async () => {
      mockApi.get.mockRejectedValueOnce('string error');

      await useOperationsStore.getState().fetch();

      expect(useOperationsStore.getState().error).toBe('Failed to fetch operations');
    });
  });

  describe('cancel', () => {
    it('cancels an operation and refreshes', async () => {
      mockApi.post.mockResolvedValueOnce({});
      mockApi.get.mockResolvedValueOnce({operations: [], counts: {}});

      const result = await useOperationsStore.getState().cancel('op1');

      expect(result).toBe(true);
      expect(mockApi.post).toHaveBeenCalledWith('/cancel/op1');
    });

    it('returns false on failure', async () => {
      mockApi.post.mockRejectedValueOnce(new Error('fail'));

      const result = await useOperationsStore.getState().cancel('op1');

      expect(result).toBe(false);
    });
  });

  describe('cancelAll', () => {
    it('cancels all operations', async () => {
      mockApi.post.mockResolvedValueOnce({cancelled: ['op1', 'op2'], count: 2});
      mockApi.get.mockResolvedValueOnce({operations: [], counts: {}});

      const count = await useOperationsStore.getState().cancelAll();

      expect(count).toBe(2);
      expect(mockApi.post).toHaveBeenCalledWith('/cancel-all');
    });

    it('cancels operations by type', async () => {
      mockApi.post.mockResolvedValueOnce({cancelled: ['op1'], count: 1});
      mockApi.get.mockResolvedValueOnce({operations: [], counts: {}});

      const count = await useOperationsStore.getState().cancelAll('training');

      expect(count).toBe(1);
      expect(mockApi.post).toHaveBeenCalledWith('/cancel-all?type=training');
    });

    it('returns 0 on failure', async () => {
      mockApi.post.mockRejectedValueOnce(new Error('fail'));

      const count = await useOperationsStore.getState().cancelAll();

      expect(count).toBe(0);
    });
  });

  describe('startPolling', () => {
    beforeEach(() => {
      jest.useFakeTimers();
    });

    afterEach(() => {
      jest.useRealTimers();
    });

    it('starts polling and calls fetch immediately', () => {
      mockApi.get.mockResolvedValue({operations: [], counts: {}});

      useOperationsStore.getState().startPolling(5000);

      expect(mockApi.get).toHaveBeenCalled();
      expect(useOperationsStore.getState()._pollTimer).not.toBeNull();
    });

    it('clears existing timer before creating new one', () => {
      mockApi.get.mockResolvedValue({operations: [], counts: {}});
      const spy = jest.spyOn(global, 'clearInterval');

      useOperationsStore.getState().startPolling(5000);
      useOperationsStore.getState().startPolling(3000);

      expect(spy).toHaveBeenCalled();
      spy.mockRestore();
    });
  });

  describe('stopPolling', () => {
    it('clears the polling timer', () => {
      mockApi.get.mockResolvedValue({operations: [], counts: {}});
      useOperationsStore.getState().startPolling(5000);

      useOperationsStore.getState().stopPolling();

      expect(useOperationsStore.getState()._pollTimer).toBeNull();
    });

    it('is safe to call when no timer', () => {
      expect(() => useOperationsStore.getState().stopPolling()).not.toThrow();
    });
  });

  describe('activeByType', () => {
    it('returns active operations of given type', () => {
      useOperationsStore.setState({
        operations: [
          makeOp({id: 'op1', type: 'training', status: 'running'}),
          makeOp({id: 'op2', type: 'training', status: 'completed'}),
          makeOp({id: 'op3', type: 'inference', status: 'running'}),
        ],
      });

      const active = useOperationsStore.getState().activeByType('training');
      expect(active).toHaveLength(1);
      expect(active[0].id).toBe('op1');
    });

    it('includes registered and cancelling statuses', () => {
      useOperationsStore.setState({
        operations: [
          makeOp({id: 'op1', type: 'training', status: 'registered'}),
          makeOp({id: 'op2', type: 'training', status: 'cancelling'}),
          makeOp({id: 'op3', type: 'training', status: 'cancelled'}),
        ],
      });

      const active = useOperationsStore.getState().activeByType('training');
      expect(active).toHaveLength(2);
    });
  });

  describe('isAnyActive', () => {
    it('returns true when any active operations exist', () => {
      useOperationsStore.setState({
        operations: [makeOp({status: 'running'})],
      });
      expect(useOperationsStore.getState().isAnyActive()).toBe(true);
    });

    it('returns false when no active operations', () => {
      useOperationsStore.setState({
        operations: [makeOp({status: 'completed'}), makeOp({status: 'failed'})],
      });
      expect(useOperationsStore.getState().isAnyActive()).toBe(false);
    });

    it('filters by type when provided', () => {
      useOperationsStore.setState({
        operations: [
          makeOp({id: 'op1', type: 'training', status: 'running'}),
          makeOp({id: 'op2', type: 'inference', status: 'completed'}),
        ],
      });
      expect(useOperationsStore.getState().isAnyActive('training')).toBe(true);
      expect(useOperationsStore.getState().isAnyActive('download')).toBe(false);
    });
  });

  describe('hasActive', () => {
    it('delegates to isAnyActive with type', () => {
      useOperationsStore.setState({
        operations: [makeOp({type: 'download', status: 'registered'})],
      });
      expect(useOperationsStore.getState().hasActive('download')).toBe(true);
      expect(useOperationsStore.getState().hasActive('training')).toBe(false);
    });
  });
});
