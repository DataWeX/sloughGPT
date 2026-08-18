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
  },
}));

jest.mock('../../services/haptics', () => ({
  triggerHaptic: jest.fn(),
}));

jest.mock('../../services/toast', () => ({
  toast: {success: jest.fn(), error: jest.fn(), info: jest.fn()},
}));

jest.mock('../../components/StatusBadge', () => ({
  StatusBadge: ({label, variant}: any) => {
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
const {FeedbackScreen} = require('../FeedbackScreen');

describe('FeedbackScreen', () => {
  it('renders title', async () => {
    const {getByText} = await render(<FeedbackScreen />);
    await waitFor(() => {
      expect(getByText('Feedback')).toBeTruthy();
    });
  });

  it('shows stats cards', async () => {
    const mockApi = require('../../services/api-client').api;
    mockApi.get
      .mockResolvedValueOnce({total: 100, positive: 80, negative: 20})
      .mockResolvedValueOnce({active: true, pending_count: 5, completed_count: 95, last_run: null})
      .mockResolvedValueOnce({total_pairs: 50, synced: 40, pending: 10});
    const {getByText} = await render(<FeedbackScreen />);
    await waitFor(() => {
      expect(getByText('100')).toBeTruthy();
      expect(getByText('80')).toBeTruthy();
      expect(getByText('20')).toBeTruthy();
    });
  });

  it('shows Workflow section', async () => {
    const {getByText} = await render(<FeedbackScreen />);
    await waitFor(() => {
      expect(getByText('Workflow')).toBeTruthy();
    });
  });

  it('shows Aggregate button', async () => {
    const {getByText} = await render(<FeedbackScreen />);
    await waitFor(() => {
      expect(getByText('Aggregate')).toBeTruthy();
    });
  });

  it('shows Evaluate button', async () => {
    const {getByText} = await render(<FeedbackScreen />);
    await waitFor(() => {
      expect(getByText('Evaluate')).toBeTruthy();
    });
  });
});
