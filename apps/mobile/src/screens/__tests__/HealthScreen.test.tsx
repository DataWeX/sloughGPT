import React from 'react';
import {render, fireEvent, waitFor} from '@/test-utils';

// ── Module-level mocks ──────────────────────────────────────────────────

const mockNavigate = jest.fn();

jest.mock('react-native-safe-area-context', () => {
  const React = require('react');
  const {View} = require('react-native');
  return {
    SafeAreaView: ({children, edges, style}: any) =>
      React.createElement(View, {style, testID: 'safe-area-view'}, children),
    useSafeAreaInsets: () => ({top: 0, bottom: 0, left: 0, right: 0}),
  };
});

jest.mock('../../services/api-client', () => ({
  api: {
    get: jest.fn().mockRejectedValue(new Error('default mock')),
  },
}));

jest.mock('../../components/StatusBadge', () => ({
  StatusBadge: ({label, variant}: any) => {
    const React = require('react');
    const {Text} = require('react-native');
    return React.createElement(Text, {testID: `status-badge-${variant}`}, label);
  },
}));

import {api} from '../../services/api-client';
const mockApiGet = api.get as jest.Mock;

// ── Default mock health data ────────────────────────────────────────────

const detailedHealth: any = {
  api: {status: 'healthy', model_loaded: true, model_name: 'gpt2'},
  system: {
    cpu_percent: 45.2,
    memory_percent: 62.3,
    memory_used_gb: 4.8,
    memory_total_gb: 8.0,
    disk_used_gb: 120.5,
    disk_free_gb: 256.3,
    disk_total_gb: 376.8,
    uptime: 86400,
  },
};

beforeEach(() => {
  jest.clearAllMocks();
  jest.useFakeTimers();
});

afterEach(() => {
  jest.runOnlyPendingTimers();
  jest.useRealTimers();
});

// eslint-disable-next-line @typescript-eslint/no-var-requires
const {HealthScreen} = require('../HealthScreen');

describe('HealthScreen', () => {
  it('renders title and API status on load', async () => {
    mockApiGet.mockResolvedValue(detailedHealth);
    const {getByText} = await render(<HealthScreen />);
    await waitFor(() => {
      expect(getByText('System Health')).toBeTruthy();
      expect(getByText('Healthy')).toBeTruthy();
    });
  });

  it('shows Loaded badge when model is loaded', async () => {
    mockApiGet.mockResolvedValue(detailedHealth);
    const {getByText} = await render(<HealthScreen />);
    await waitFor(() => {
      expect(getByText('Loaded')).toBeTruthy();
    });
  });

  it('shows active model name', async () => {
    mockApiGet.mockResolvedValue(detailedHealth);
    const {getByText} = await render(<HealthScreen />);
    await waitFor(() => {
      expect(getByText('gpt2')).toBeTruthy();
    });
  });

  it('shows CPU percent', async () => {
    mockApiGet.mockResolvedValue(detailedHealth);
    const {getByText} = await render(<HealthScreen />);
    await waitFor(() => {
      expect(getByText('45.2%')).toBeTruthy();
    });
  });

  it('shows memory used/total', async () => {
    mockApiGet.mockResolvedValue(detailedHealth);
    const {getByText} = await render(<HealthScreen />);
    await waitFor(() => {
      expect(getByText(/4\.8.*8\.0/)).toBeTruthy();
    });
  });

  it('shows disk used/total', async () => {
    mockApiGet.mockResolvedValue(detailedHealth);
    const {getByText} = await render(<HealthScreen />);
    await waitFor(() => {
      expect(getByText(/120\.5.*376\.8/)).toBeTruthy();
    });
  });

  it('formats uptime > 24h correctly', async () => {
    mockApiGet.mockResolvedValue(detailedHealth);
    const {getByText} = await render(<HealthScreen />);
    await waitFor(() => {
      expect(getByText('24h 0m')).toBeTruthy();
    });
  });

  it('handles API failure by falling back to basic health', async () => {
    mockApiGet
      .mockRejectedValueOnce(new Error('detailed fail'))
      .mockResolvedValueOnce({status: 'healthy', model_loaded: false, uptime: 300});
    const {getByText, queryByText} = await render(<HealthScreen />);
    await waitFor(() => {
      expect(getByText('Healthy')).toBeTruthy();
    });
    expect(queryByText('Loaded')).toBeNull();
  });

  it('handles total API failure gracefully', async () => {
    mockApiGet.mockRejectedValue(new Error('network error'));
    const {getByText} = await render(<HealthScreen />);
    await waitFor(() => {
      expect(getByText('System Health')).toBeTruthy();
    });
  });

  it('cleans up interval on unmount', async () => {
    const clearSpy = jest.spyOn(global, 'clearInterval');
    mockApiGet.mockResolvedValue(detailedHealth);
    const {unmount} = await render(<HealthScreen />);
    await unmount();
    expect(clearSpy).toHaveBeenCalled();
  });
});
