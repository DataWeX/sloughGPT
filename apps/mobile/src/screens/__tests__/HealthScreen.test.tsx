import React from 'react';
import {render, fireEvent, waitFor} from '@/test-utils';

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

jest.mock('../../components/Icon', () => ({
  Icon: ({name, size, color}: any) => {
    const React = require('react');
    const {Text} = require('react-native');
    return React.createElement(Text, {}, '');
  },
}));

import {api} from '../../services/api-client';
const mockApiGet = api.get as jest.Mock;

const detailedHealth: any = {
  status: 'healthy',
  uptime_seconds: 86400,
  timestamp: new Date().toISOString(),
  request_count: 150,
  error_count: 2,
  avg_latency_ms: 120,
  requests_per_minute: 5.2,
  inference_count: 50,
  total_tokens: 12000,
  tokens_per_sec: 12.5,
  avg_tokens_per_request: 240,
  model_loaded: true,
  model_loading: false,
  model_type: 'gpt2',
  device: 'cpu',
  soul: 'friendly',
  system: {
    cpu_percent: 45.2,
    memory_percent: 62.3,
    memory_available_mb: 3072,
    open_files: 128,
    threads: 24,
    rss_mb: 2048,
  },
  gpu: {backend: 'unknown'},
  health_score: {score: 85, status: 'healthy', diagnoses: []},
  kv_sessions: {enabled: true, active_sessions: 2, cached_tokens: 1500, ttl_seconds: 300},
  training_pool: {active_jobs: 0, max_workers: 2, total_tracked: 5},
  lifecycle: {phase: 'running', is_running: true, in_flight: 3},
  recent_errors: [],
  status_message: 'Healthy',
};

beforeEach(() => {
  jest.clearAllMocks();
  jest.useFakeTimers();
});

afterEach(() => {
  jest.runOnlyPendingTimers();
  jest.useRealTimers();
});

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

  it('shows model type', async () => {
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

  it('shows memory percent', async () => {
    mockApiGet.mockResolvedValue(detailedHealth);
    const {getByText} = await render(<HealthScreen />);
    await waitFor(() => {
      expect(getByText('62.3%')).toBeTruthy();
    });
  });

  it('shows health score', async () => {
    mockApiGet.mockResolvedValue(detailedHealth);
    const {getByText} = await render(<HealthScreen />);
    await waitFor(() => {
      expect(getByText('85')).toBeTruthy();
    });
  });

  it('shows soul name', async () => {
    mockApiGet.mockResolvedValue(detailedHealth);
    const {getByText} = await render(<HealthScreen />);
    await waitFor(() => {
      expect(getByText('friendly')).toBeTruthy();
    });
  });

  it('formats uptime > 24h correctly', async () => {
    mockApiGet.mockResolvedValue(detailedHealth);
    const {getByText} = await render(<HealthScreen />);
    await waitFor(() => {
      expect(getByText('1d 0h')).toBeTruthy();
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
