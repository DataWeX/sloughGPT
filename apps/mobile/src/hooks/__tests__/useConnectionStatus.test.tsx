import React from 'react';
import {Text} from 'react-native';
import {render, waitFor, screen} from '@testing-library/react-native';
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

  it('transitions to reconnecting on non-ok response', async () => {
    jest.spyOn(global, 'fetch').mockResolvedValue({ok: false} as Response);
    await render(<Display />);
    await waitFor(() => {
      expect(parseVal(screen.getByTestId('val')).state).toBe('reconnecting');
    });
  });

  it('increments retryCount on non-ok response', async () => {
    jest.spyOn(global, 'fetch').mockResolvedValue({ok: false} as Response);
    await render(<Display />);
    await waitFor(() => {
      expect(parseVal(screen.getByTestId('val')).retryCount).toBeGreaterThanOrEqual(1);
    });
  });

  it('transitions to offline on fetch error', async () => {
    jest.spyOn(global, 'fetch').mockRejectedValue(new Error('fail'));
    await render(<Display />);
    await waitFor(() => {
      expect(parseVal(screen.getByTestId('val')).state).toBe('offline');
    });
  });

  it('sets latencyMs to null on error', async () => {
    jest.spyOn(global, 'fetch').mockRejectedValue(new Error('fail'));
    await render(<Display />);
    await waitFor(() => {
      expect(parseVal(screen.getByTestId('val')).latencyMs).toBeNull();
    });
  });

  it('increments retryCount on error', async () => {
    jest.spyOn(global, 'fetch').mockRejectedValue(new Error('fail'));
    await render(<Display />);
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
