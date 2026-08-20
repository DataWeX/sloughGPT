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
const {ErrorsScreen} = require('../ErrorsScreen');

describe('ErrorsScreen', () => {
  it('renders title', async () => {
    const {getByText} = await render(<ErrorsScreen />);
    await waitFor(() => {
      expect(getByText('Errors')).toBeTruthy();
    });
  });

  it('shows grouped/recent tabs', async () => {
    const {getByText} = await render(<ErrorsScreen />);
    await waitFor(() => {
      expect(getByText('Grouped')).toBeTruthy();
      expect(getByText('Recent')).toBeTruthy();
    });
  });

  it('shows empty state when no errors', async () => {
    const {getByText} = await render(<ErrorsScreen />);
    await waitFor(() => {
      expect(getByText('No errors logged')).toBeTruthy();
    });
  });

  it('loads grouped errors from API', async () => {
    const mockApi = require('../../services/api-client').api;
    mockApi.get
      .mockResolvedValueOnce({
        groups: [
          {fingerprint: 'abc', message: 'Test error', source: 'web', count: 5, latest: new Date().toISOString(), sample_url: '/test'},
        ],
      })
      .mockResolvedValueOnce({errors: [], total: 0});
    const {getByText} = await render(<ErrorsScreen />);
    await waitFor(() => {
      expect(getByText('Test error')).toBeTruthy();
      expect(getByText('5x')).toBeTruthy();
    });
  });

  it('shows error count in header', async () => {
    const mockApi = require('../../services/api-client').api;
    mockApi.get
      .mockResolvedValueOnce({groups: []})
      .mockResolvedValueOnce({errors: [], total: 42});
    const {getByText} = await render(<ErrorsScreen />);
    await waitFor(() => {
      expect(getByText('42 total errors')).toBeTruthy();
    });
  });
});
