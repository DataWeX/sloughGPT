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
    post: jest.fn().mockResolvedValue({results: []}),
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
const {LearnScreen} = require('../LearnScreen');

describe('LearnScreen', () => {
  it('renders title', async () => {
    const {getByText} = await render(<LearnScreen />);
    await waitFor(() => {
      expect(getByText('Learn')).toBeTruthy();
    });
  });

  it('shows search tab by default', async () => {
    const {getAllByText} = await render(<LearnScreen />);
    await waitFor(() => {
      expect(getAllByText('Search').length).toBeGreaterThanOrEqual(1);
    });
  });

  it('shows all four tabs', async () => {
    const {getAllByText} = await render(<LearnScreen />);
    await waitFor(() => {
      expect(getAllByText('Search').length).toBeGreaterThanOrEqual(1);
      expect(getAllByText('Ingest').length).toBeGreaterThanOrEqual(1);
      expect(getAllByText('Knowledge').length).toBeGreaterThanOrEqual(1);
      expect(getAllByText('Feeds').length).toBeGreaterThanOrEqual(1);
    });
  });

  it('shows stats grid', async () => {
    const {getAllByText} = await render(<LearnScreen />);
    await waitFor(() => {
      expect(getAllByText('Facts').length).toBeGreaterThanOrEqual(1);
      expect(getAllByText('Tokens').length).toBeGreaterThanOrEqual(1);
    });
  });

  it('has search input', async () => {
    const {getByPlaceholderText} = await render(<LearnScreen />);
    await waitFor(() => {
      expect(getByPlaceholderText('Search knowledge...')).toBeTruthy();
    });
  });
});
