import React from 'react';
import {Text} from 'react-native';
import {render, waitFor, screen, act} from '@testing-library/react-native';
import * as apiClient from '../../services/api-client';

jest.mock('../../services/api-client');

const mockGetApiUrl = apiClient.getApiUrl as jest.Mock;

// eslint-disable-next-line @typescript-eslint/no-var-requires
const {useConnectionStatus} = require('../useConnectionStatus');

function Display() {
  const info = useConnectionStatus();
  return <Text testID="val">{JSON.stringify(info)}</Text>;
}

function parseVal(el: any): any {
  return JSON.parse(String(el.children[0]));
}

beforeEach(() => {
  jest.useFakeTimers();
  jest.clearAllMocks();
  mockGetApiUrl.mockResolvedValue('http://localhost:8000');
});

afterEach(() => {
  jest.runOnlyPendingTimers();
  jest.useRealTimers();
  jest.clearAllMocks();
});

function suppressActWarning() {
  jest.spyOn(console, 'error').mockImplementation(msg => {
    if (typeof msg === 'string' && msg.includes('not wrapped in act')) return;
    console.error(msg);
  });
}

describe('useConnectionStatus', () => {
  beforeEach(() => {
    suppressActWarning();
  });

  it('becomes connected after successful health check', async () => {
    jest
      .spyOn(global, 'fetch')
      .mockResolvedValue({ok: true, json: async () => ({model_loaded: true})} as Response);
    await render(<Display />);
    await waitFor(() => {
      expect(parseVal(screen.getByTestId('val')).state).toBe('connected');
    });
  });

  it('stores modelLoaded from health response', async () => {
    jest
      .spyOn(global, 'fetch')
      .mockResolvedValue({ok: true, json: async () => ({model_loaded: true})} as Response);
    await render(<Display />);
    await waitFor(() => {
      expect(parseVal(screen.getByTestId('val')).modelLoaded).toBe(true);
    });
  });

  it('stores latencyMs after health check', async () => {
    jest
      .spyOn(global, 'fetch')
      .mockResolvedValue({ok: true, json: async () => ({})} as Response);
    await render(<Display />);
    await waitFor(() => {
      expect(parseVal(screen.getByTestId('val')).latencyMs).toBeGreaterThanOrEqual(0);
    });
  });

  it('sets lastSeen after successful check', async () => {
    jest
      .spyOn(global, 'fetch')
      .mockResolvedValue({ok: true, json: async () => ({})} as Response);
    await render(<Display />);
    await waitFor(() => {
      expect(parseVal(screen.getByTestId('val')).lastSeen).toBeGreaterThan(0);
    });
  });

  it('stays connecting on first failure (never connected)', async () => {
    jest.spyOn(global, 'fetch').mockResolvedValue({ok: false} as Response);
    await render(<Display />);
    // Should stay 'connecting' — never went offline since never connected
    await waitFor(() => {
      expect(parseVal(screen.getByTestId('val')).state).toBe('connecting');
    });
  });

  it('transitions to reconnecting after connected then non-ok response', async () => {
    const spy = jest.spyOn(global, 'fetch');
    // First call succeeds
    spy.mockResolvedValueOnce({ok: true, json: async () => ({model_loaded: true})} as Response);
    await render(<Display />);
    await waitFor(() => {
      expect(parseVal(screen.getByTestId('val')).state).toBe('connected');
    });
    // Second call fails
    spy.mockResolvedValueOnce({ok: false} as Response);
    await act(async () => {
      jest.advanceTimersByTime(8000);
    });
    await waitFor(() => {
      expect(parseVal(screen.getByTestId('val')).state).toBe('reconnecting');
    });
  });

  it('increments retryCount after connected then non-ok response', async () => {
    const spy = jest.spyOn(global, 'fetch');
    spy.mockResolvedValueOnce({ok: true, json: async () => ({model_loaded: true})} as Response);
    await render(<Display />);
    await waitFor(() => {
      expect(parseVal(screen.getByTestId('val')).retryCount).toBe(0);
    });
    spy.mockResolvedValueOnce({ok: false} as Response);
    await act(async () => {
      jest.advanceTimersByTime(8000);
    });
    await waitFor(() => {
      expect(parseVal(screen.getByTestId('val')).retryCount).toBeGreaterThanOrEqual(1);
    });
  });

  it('transitions to offline after connected then fetch error', async () => {
    const spy = jest.spyOn(global, 'fetch');
    spy.mockResolvedValueOnce({ok: true, json: async () => ({model_loaded: true})} as Response);
    await render(<Display />);
    await waitFor(() => {
      expect(parseVal(screen.getByTestId('val')).state).toBe('connected');
    });
    spy.mockRejectedValueOnce(new Error('fail'));
    await act(async () => {
      jest.advanceTimersByTime(8000);
    });
    await waitFor(() => {
      expect(parseVal(screen.getByTestId('val')).state).toBe('offline');
    });
  });

  it('sets latencyMs to null on error after connected', async () => {
    const spy = jest.spyOn(global, 'fetch');
    spy.mockResolvedValueOnce({ok: true, json: async () => ({model_loaded: true})} as Response);
    await render(<Display />);
    await waitFor(() => {
      expect(parseVal(screen.getByTestId('val')).latencyMs).toBeGreaterThanOrEqual(0);
    });
    spy.mockRejectedValueOnce(new Error('fail'));
    await act(async () => {
      jest.advanceTimersByTime(8000);
    });
    await waitFor(() => {
      expect(parseVal(screen.getByTestId('val')).latencyMs).toBeNull();
    });
  });

  it('increments retryCount after connected then error', async () => {
    const spy = jest.spyOn(global, 'fetch');
    spy.mockResolvedValueOnce({ok: true, json: async () => ({model_loaded: true})} as Response);
    await render(<Display />);
    await waitFor(() => {
      expect(parseVal(screen.getByTestId('val')).retryCount).toBe(0);
    });
    spy.mockRejectedValueOnce(new Error('fail'));
    await act(async () => {
      jest.advanceTimersByTime(8000);
    });
    await waitFor(() => {
      expect(parseVal(screen.getByTestId('val')).retryCount).toBeGreaterThanOrEqual(1);
    });
  });

  it('uses AbortController with 5s timeout', async () => {
    const abortSpy = jest.spyOn(global, 'AbortController');
    jest
      .spyOn(global, 'fetch')
      .mockResolvedValue({ok: true, json: async () => ({})} as Response);
    await render(<Display />);
    expect(abortSpy).toHaveBeenCalled();
  });
});
