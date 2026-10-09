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
const {MemoryScreen} = require('../MemoryScreen');

describe('MemoryScreen', () => {
  it('renders title', async () => {
    const {getByText} = await render(<MemoryScreen />);
    await waitFor(() => {
      expect(getByText('Memory')).toBeTruthy();
    });
  });

  it('shows tabs', async () => {
    const {getByText} = await render(<MemoryScreen />);
    await waitFor(() => {
      expect(getByText('All Memories')).toBeTruthy();
      expect(getByText('Store New')).toBeTruthy();
    });
  });

  it('shows empty state', async () => {
    const {getByText} = await render(<MemoryScreen />);
    await waitFor(() => {
      expect(getByText('No memories yet')).toBeTruthy();
    });
  });

  it('loads memories from API', async () => {
    const mockApi = require('../../services/api-client').api;
    mockApi.get
      .mockResolvedValueOnce({
        items: [
          {id: '1', content: 'User prefers dark mode', topic: 'preferences', importance: 0.8, source: 'chat', created_at: new Date().toISOString()},
        ],
      })
      .mockResolvedValueOnce({total_items: 1, topics: ['preferences'], enabled: true});
    const {getByText} = await render(<MemoryScreen />);
    await waitFor(() => {
      expect(getByText('User prefers dark mode')).toBeTruthy();
      expect(getByText('preferences')).toBeTruthy();
    });
  });

  it('shows stats when loaded', async () => {
    const mockApi = require('../../services/api-client').api;
    mockApi.get
      .mockResolvedValueOnce({items: []})
      .mockResolvedValueOnce({total_items: 15, topics: ['prefs', 'work', 'name'], enabled: true});
    const {getByText} = await render(<MemoryScreen />);
    await waitFor(() => {
      expect(getByText('15')).toBeTruthy();
      expect(getByText('3')).toBeTruthy();
      expect(getByText('Enabled')).toBeTruthy();
    });
  });
});
