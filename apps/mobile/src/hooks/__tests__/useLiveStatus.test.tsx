import React from 'react';
import {Text} from 'react-native';
import {render, waitFor, screen, act} from '@testing-library/react-native';
import {api} from '../../services/api-client';

jest.mock('../../services/api-client');

const mockApi = api as jest.Mocked<typeof api>;

// eslint-disable-next-line @typescript-eslint/no-var-requires
const {useLiveStatus} = require('../useLiveStatus');

function Display({enabled, interval}: {enabled?: boolean; interval?: number}) {
  const {health, connectionStatus} = useLiveStatus({enabled, pollIntervalMs: interval});
  return (
    <Text testID="val">
      {JSON.stringify({
        status: connectionStatus,
        loaded: health?.model_loaded ?? null,
        soul: health?.soul ?? null,
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
  mockApi.get.mockResolvedValue({
    status: 'ok',
    model_loaded: true,
    model_loading: false,
    model_type: 'slnet',
    soul: 'default',
    uptime: 100,
    inference_count: 5,
    total_tokens: 1000,
    tokens_per_sec: 50,
    avg_latency_ms: 20,
    cpu_percent: 30,
    memory_percent: 60,
    memory_used_gb: 4,
    memory_total_gb: 8,
    request_count: 10,
    error_count: 0,
    requests_per_minute: 2,
    health_score: 95,
    status_message: 'Healthy',
  });
});

afterEach(() => {
  jest.useRealTimers();
});

describe('useLiveStatus', () => {
  it('fetches health on mount', async () => {
    await render(<Display />);
    expect(mockApi.get).toHaveBeenCalledWith('/health');
  });

  it('updates to connected on successful fetch', async () => {
    await render(<Display />);

    await waitFor(() => {
      const val = parseVal();
      expect(val.status).toBe('connected');
      expect(val.loaded).toBe(true);
    });
  });

  it('updates to disconnected on failure', async () => {
    mockApi.get.mockRejectedValue(new Error('network'));

    await render(<Display />);

    await waitFor(() => {
      expect(parseVal().status).toBe('disconnected');
    });
  });

  it('does not fetch when disabled', async () => {
    await render(<Display enabled={false} />);
    expect(mockApi.get).not.toHaveBeenCalled();
  });

  it('polls at specified interval', async () => {
    await render(<Display interval={1000} />);

    await act(async () => {
      jest.advanceTimersByTime(1000);
    });

    expect(mockApi.get).toHaveBeenCalledTimes(2);
  });
});
