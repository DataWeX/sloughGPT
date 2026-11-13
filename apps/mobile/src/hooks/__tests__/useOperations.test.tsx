import React from 'react';
import {Text} from 'react-native';
import {render, waitFor, screen, act} from '@testing-library/react-native';
import {useOperationsStore} from '../../stores/operations-store';
import {api} from '../../services/api-client';

jest.mock('../../services/api-client');

const mockApi = api as jest.Mocked<typeof api>;

// eslint-disable-next-line @typescript-eslint/no-var-requires
const {useOperations} = require('../useOperations');

function Display({type, interval}: {type?: string; interval?: number}) {
  const ops = useOperations(type, interval);
  return (
    <Text testID="val">
      {JSON.stringify({
        opCount: ops.operations.length,
        activeCount: ops.activeOps.length,
        loading: ops.loading,
        error: ops.error,
        isActive: ops.isActive,
        hasTraining: ops.hasTraining,
        hasInference: ops.hasInference,
      })}
    </Text>
  );
}

function parseVal(): any {
  return JSON.parse(String(screen.getByTestId('val').children[0]));
}

beforeEach(() => {
  jest.clearAllMocks();
  jest.useFakeTimers();
  useOperationsStore.setState({
    operations: [],
    counts: {},
    loading: false,
    error: null,
    _pollTimer: null,
  });
  mockApi.get.mockResolvedValue({operations: [], counts: {}});
});

afterEach(() => {
  useOperationsStore.getState().stopPolling();
  jest.useRealTimers();
});

describe('useOperations', () => {
  it('returns initial state', async () => {
    await render(<Display />);
    const val = parseVal();
    expect(val.opCount).toBe(0);
    expect(val.activeCount).toBe(0);
    expect(val.loading).toBe(false);
    expect(val.error).toBeNull();
    expect(val.isActive).toBe(false);
    expect(val.hasTraining).toBe(false);
    expect(val.hasInference).toBe(false);
  });

  it('starts polling on mount', async () => {
    await render(<Display />);
    expect(mockApi.get).toHaveBeenCalled();
  });

  it('polls at specified interval', async () => {
    await render(<Display interval={1000} />);

    await act(async () => {
      jest.advanceTimersByTime(1000);
    });

    expect(mockApi.get).toHaveBeenCalledTimes(2);
  });
});
