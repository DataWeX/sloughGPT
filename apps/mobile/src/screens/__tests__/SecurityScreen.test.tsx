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
  },
}));

jest.mock('../../services/haptics', () => ({
  triggerHaptic: jest.fn(),
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
const {SecurityScreen} = require('../SecurityScreen');

describe('SecurityScreen', () => {
  it('renders title', async () => {
    const {getByText} = await render(<SecurityScreen />);
    await waitFor(() => {
      expect(getByText('Security')).toBeTruthy();
    });
  });

  it('shows API Keys section', async () => {
    const {getByText} = await render(<SecurityScreen />);
    await waitFor(() => {
      expect(getByText('API Keys')).toBeTruthy();
    });
  });

  it('loads audit logs from API', async () => {
    const mockApi = require('../../services/api-client').api;
    mockApi.get
      .mockResolvedValueOnce({
        logs: [
          {event_type: 'api_key.created', timestamp: new Date().toISOString(), user: 'admin', resource: 'key-1'},
        ],
        count: 1,
      })
      .mockResolvedValueOnce({count: 2, configured: true});
    const {getAllByText} = await render(<SecurityScreen />);
    await waitFor(() => {
      expect(getAllByText('api_key.created').length).toBeGreaterThanOrEqual(1);
      expect(getAllByText('admin').length).toBeGreaterThanOrEqual(1);
    });
  });

  it('shows configured status for API keys', async () => {
    const mockApi = require('../../services/api-client').api;
    mockApi.get
      .mockResolvedValueOnce({logs: [], count: 0})
      .mockResolvedValueOnce({count: 3, configured: true});
    const {getByText} = await render(<SecurityScreen />);
    await waitFor(() => {
      expect(getByText('3 configured')).toBeTruthy();
    });
  });

  it('shows empty state when no audit events', async () => {
    const {getByText} = await render(<SecurityScreen />);
    await waitFor(() => {
      expect(getByText('No audit events')).toBeTruthy();
    });
  });
});
