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
    get: jest.fn().mockResolvedValue({adapters: []}),
    post: jest.fn().mockResolvedValue({verdict: 'merged'}),
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
const {AdaptersScreen} = require('../AdaptersScreen');

describe('AdaptersScreen', () => {
  it('renders title', async () => {
    const {getByText} = await render(<AdaptersScreen />);
    await waitFor(() => {
      expect(getByText('Adapters')).toBeTruthy();
    });
  });

  it('shows Aggregate button', async () => {
    const {getByText} = await render(<AdaptersScreen />);
    await waitFor(() => {
      expect(getByText('Aggregate')).toBeTruthy();
    });
  });

  it('shows empty state when no adapters', async () => {
    const {getByText} = await render(<AdaptersScreen />);
    await waitFor(() => {
      expect(getByText('No adapters')).toBeTruthy();
    });
  });

  it('renders adapter list when data exists', async () => {
    const mockApi = require('../../services/api-client').api;
    mockApi.get.mockResolvedValueOnce({
      adapters: [
        {id: '1', user_id: 'user1', name: 'my-adapter', loss: 0.5, steps: 100, created_at: '2024-01-01'},
      ],
    });
    const {getByText} = await render(<AdaptersScreen />);
    await waitFor(() => {
      expect(getByText('my-adapter')).toBeTruthy();
      expect(getByText('loss: 0.5000')).toBeTruthy();
    });
  });
});
