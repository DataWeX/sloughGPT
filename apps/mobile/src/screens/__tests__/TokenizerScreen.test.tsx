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
    get: jest.fn(),
    post: jest.fn(),
  },
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

jest.mock('react-native', () => {
  const RN = jest.requireActual('react-native');
  RN.Clipboard = {setString: jest.fn()};
  return RN;
});

import {api} from '../../services/api-client';
const mockApiGet = api.get as jest.Mock;

beforeEach(() => {
  jest.clearAllMocks();
  // Default: all GET calls return null (stats + samples + tokenize)
  mockApiGet.mockResolvedValue(null);
});

// eslint-disable-next-line @typescript-eslint/no-var-requires
const {TokenizerScreen} = require('../TokenizerScreen');

describe('TokenizerScreen', () => {
  it('renders title', async () => {
    const {getByText} = await render(<TokenizerScreen />);
    await waitFor(() => {
      expect(getByText('Tokenizer')).toBeTruthy();
    });
  });

  it('renders playground section', async () => {
    const {getByText} = await render(<TokenizerScreen />);
    await waitFor(() => {
      expect(getByText('Playground')).toBeTruthy();
    });
  });

  it('shows stats when available', async () => {
    mockApiGet
      .mockResolvedValueOnce({vocab_size: 50257, total_merges: 50000, model_name: 'gpt2'})
      .mockResolvedValueOnce([]);
    const {getByText} = await render(<TokenizerScreen />);
    await waitFor(() => {
      expect(getByText('50,257')).toBeTruthy();
      expect(getByText('Vocab Size')).toBeTruthy();
    });
  });

  it('handles API failure gracefully', async () => {
    // Stats returns null, samples returns [] — both handled by .catch()
    mockApiGet.mockResolvedValue(null);
    const {getByText} = await render(<TokenizerScreen />);
    await waitFor(() => {
      // Component renders even with null data — no crash
      expect(getByText('Tokenizer')).toBeTruthy();
      expect(getByText('Playground')).toBeTruthy();
    });
  });
});
