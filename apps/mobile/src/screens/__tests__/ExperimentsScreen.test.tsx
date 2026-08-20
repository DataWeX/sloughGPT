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
    get: jest.fn().mockResolvedValue(null),
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
const {ExperimentsScreen} = require('../ExperimentsScreen');

describe('ExperimentsScreen', () => {
  it('renders title', async () => {
    const {getByText} = await render(<ExperimentsScreen />);
    await waitFor(() => {
      expect(getByText('Experiments')).toBeTruthy();
    });
  });

  it('shows create form', async () => {
    const {getByText} = await render(<ExperimentsScreen />);
    await waitFor(() => {
      expect(getByText('New Experiment')).toBeTruthy();
    });
  });

  it('shows empty state', async () => {
    const {getByText} = await render(<ExperimentsScreen />);
    await waitFor(() => {
      expect(getByText('No experiments yet')).toBeTruthy();
    });
  });

  it('loads experiments from API', async () => {
    const mockApi = require('../../services/api-client').api;
    mockApi.get.mockResolvedValueOnce({experiments: ['exp-1', 'exp-2']});
    const {getByText} = await render(<ExperimentsScreen />);
    await waitFor(() => {
      expect(getByText('exp-1')).toBeTruthy();
      expect(getByText('exp-2')).toBeTruthy();
    });
  });

  it('shows experiment count', async () => {
    const mockApi = require('../../services/api-client').api;
    mockApi.get.mockResolvedValueOnce({experiments: ['a', 'b', 'c']});
    const {getByText} = await render(<ExperimentsScreen />);
    await waitFor(() => {
      expect(getByText('3 experiments')).toBeTruthy();
    });
  });
});
