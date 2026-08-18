import React from 'react';
import {render, waitFor} from '@/test-utils';

jest.mock('@react-navigation/native', () => ({
  useNavigation: () => ({navigate: jest.fn()}),
}));

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
    get: jest.fn().mockResolvedValue({datasets: []}),
    post: jest.fn().mockResolvedValue(null),
    delete: jest.fn().mockResolvedValue(null),
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
const {DatasetsScreen} = require('../DatasetsScreen');

describe('DatasetsScreen', () => {
  it('renders title', async () => {
    const {getByText} = await render(<DatasetsScreen />);
    await waitFor(() => {
      expect(getByText('Datasets')).toBeTruthy();
    });
  });

  it('shows empty state when no datasets', async () => {
    const {getByText} = await render(<DatasetsScreen />);
    await waitFor(() => {
      expect(getByText('No datasets')).toBeTruthy();
    });
  });

  it('shows Import button', async () => {
    const {getByText} = await render(<DatasetsScreen />);
    await waitFor(() => {
      expect(getByText('Import')).toBeTruthy();
    });
  });

  it('shows URL button', async () => {
    const {getByText} = await render(<DatasetsScreen />);
    await waitFor(() => {
      expect(getByText('URL')).toBeTruthy();
    });
  });

  it('renders dataset list when data exists', async () => {
    const mockApi = require('../../services/api-client').api;
    mockApi.get.mockResolvedValueOnce({
      datasets: [
        {id: '1', name: 'shakespeare', format: 'txt', rows: 1000, size: 50000, description: 'Shakespeare corpus'},
      ],
    });
    const {getByText} = await render(<DatasetsScreen />);
    await waitFor(() => {
      expect(getByText('shakespeare')).toBeTruthy();
      expect(getByText('txt')).toBeTruthy();
    });
  });
});
