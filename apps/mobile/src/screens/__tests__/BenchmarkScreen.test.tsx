import React from 'react';
import {render, waitFor} from '@/test-utils';

jest.mock('react-native-safe-area-context', () => {
  const React = require('react');
  const {View} = require('react-native');
  return {
    SafeAreaView: ({children, style}: any) =>
      React.createElement(View, {style, testID: 'safe-area-view'}, children),
    useSafeAreaInsets: () => ({top: 0, bottom: 0, left: 0, right: 0}),
  };
});

jest.mock('../../services/api-client', () => ({
  api: {
    get: jest.fn().mockResolvedValue({results: []}),
    post: jest.fn().mockResolvedValue({status: 'ok'}),
  },
}));

jest.mock('../../services/haptics', () => ({
  triggerHaptic: jest.fn(),
}));

jest.mock('../../services/toast', () => ({
  toast: {success: jest.fn(), error: jest.fn(), info: jest.fn()},
}));

jest.mock('../../components/StatusBadge', () => ({
  StatusBadge: ({label}: any) => {
    const React = require('react');
    const {Text} = require('react-native');
    return React.createElement(Text, null, label);
  },
}));

jest.mock('../../components/Icon', () => ({
  Icon: ({name}: any) => {
    const React = require('react');
    const {Text} = require('react-native');
    return React.createElement(Text, null, name);
  },
}));

beforeEach(() => {
  jest.clearAllMocks();
});

// eslint-disable-next-line @typescript-eslint/no-var-requires
const {BenchmarkScreen} = require('../BenchmarkScreen');

describe('BenchmarkScreen', () => {
  it('renders title', async () => {
    const {getByText} = await render(<BenchmarkScreen />);
    await waitFor(() => {
      expect(getByText('Benchmarks')).toBeTruthy();
    });
  });

  it('shows Run button', async () => {
    const {getByText} = await render(<BenchmarkScreen />);
    await waitFor(() => {
      expect(getByText('Run')).toBeTruthy();
    });
  });

  it('shows empty state when no results', async () => {
    const {getByText} = await render(<BenchmarkScreen />);
    await waitFor(() => {
      expect(getByText('No benchmark results')).toBeTruthy();
    });
  });

  it('renders benchmark results', async () => {
    const mockApi = require('../../services/api-client').api;
    mockApi.get.mockResolvedValueOnce({
      results: [
        {model: 'gpt2', coherence: 0.85, repetition: 0.12, avg_response_length: 45, perplexity: 3.2, timestamp: '2024-01-01'},
      ],
    });
    const {getByText} = await render(<BenchmarkScreen />);
    await waitFor(() => {
      expect(getByText('gpt2')).toBeTruthy();
      expect(getByText('0.85')).toBeTruthy();
    });
  });
});
