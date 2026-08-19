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

const mockRefresh = jest.fn().mockResolvedValue(undefined);
const mockModels = [
  {id: 'gpt2', name: 'GPT-2', description: '124M params', params: '124000000', size_mb: 500, source: 'huggingface'},
  {id: 'qwen', name: 'Qwen', description: '500M params', params: '500000000', size_mb: 1000, source: 'huggingface'},
];
const mockHealth = {model_name: 'gpt2'};
const mockStoreValue = {models: mockModels, health: mockHealth, refresh: mockRefresh};

jest.mock('../../services/api-client', () => ({
  api: {
    get: jest.fn().mockResolvedValue(null),
    post: jest.fn().mockResolvedValue(null),
  },
}));

jest.mock('../../stores/model-store', () => ({
  useModelStore: Object.assign(jest.fn(() => mockStoreValue), {
    getState: () => mockStoreValue,
  }),
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
const {CompareScreen} = require('../CompareScreen');

describe('CompareScreen', () => {
  it('renders title', async () => {
    const {getByText} = await render(<CompareScreen />);
    await waitFor(() => {
      expect(getByText('Compare')).toBeTruthy();
    });
  });

  it('lists available models', async () => {
    const {getByText} = await render(<CompareScreen />);
    await waitFor(() => {
      expect(getByText('GPT-2')).toBeTruthy();
      expect(getByText('Qwen')).toBeTruthy();
    });
  });

  it('shows Select Models section', async () => {
    const {getByText} = await render(<CompareScreen />);
    await waitFor(() => {
      expect(getByText('Select Models')).toBeTruthy();
    });
  });

  it('shows LOADED badge for active model', async () => {
    const {getByText} = await render(<CompareScreen />);
    await waitFor(() => {
      expect(getByText('LOADED')).toBeTruthy();
    });
  });

  it('shows model descriptions', async () => {
    const {getByText} = await render(<CompareScreen />);
    await waitFor(() => {
      expect(getByText('124M params')).toBeTruthy();
      expect(getByText('500M params')).toBeTruthy();
    });
  });
});
