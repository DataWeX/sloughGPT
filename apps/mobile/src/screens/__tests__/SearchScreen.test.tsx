/**
 * Tests for SearchScreen.
 */

import React from 'react';
import {render, fireEvent, waitFor} from '@/test-utils';

const mockLoadSession = jest.fn();

jest.mock('react-native-safe-area-context', () => {
  const React = require('react');
  const {View} = require('react-native');
  return {
    SafeAreaView: ({children}: any) =>
      React.createElement(View, {testID: 'safe-area-view'}, children),
    useSafeAreaInsets: () => ({top: 0, bottom: 0, left: 0, right: 0}),
  };
});

jest.mock('../../stores/chat-store', () => ({
  useChatStore: () => ({
    loadSession: mockLoadSession,
  }),
}));

jest.mock('../../services/api-client', () => ({
  api: {
    get: jest.fn(),
  },
}));

// eslint-disable-next-line @typescript-eslint/no-var-requires
const {SearchScreen} = require('../SearchScreen');

const mockResults = {
  results: [
    {
      id: 's1',
      name: 'Chat about AI',
      matches: [
        {role: 'user', content: 'What is AI?', timestamp: '2024-01-01'},
        {role: 'assistant', content: 'AI stands for artificial intelligence', timestamp: '2024-01-01'},
      ],
    },
    {
      id: 's2',
      name: 'Code review',
      matches: [
        {role: 'user', content: 'Review this code', timestamp: '2024-01-02'},
      ],
    },
  ],
  query: 'AI',
  total: 2,
};

describe('SearchScreen', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders without crashing', async () => {
    const view = await render(<SearchScreen />);
    expect(view.getByText('Search Messages')).toBeTruthy();
  });

  it('shows search tips when query is short', async () => {
    const view = await render(<SearchScreen />);
    expect(view.getByText(/Type at least 2 characters/)).toBeTruthy();
  });

  it('hides tips when user types', async () => {
    const {api} = require('../../services/api-client');
    api.get.mockResolvedValue({results: []});

    const view = await render(<SearchScreen />);
    const input = view.getByPlaceholderText('Search across all conversations...');
    fireEvent.changeText(input, 'AI');

    await waitFor(() => {
      expect(view.queryByText('Type at least 2 characters')).toBeNull();
    });
  });

  it('shows no results when search returns empty', async () => {
    const {api} = require('../../services/api-client');
    api.get.mockResolvedValue({results: []});

    const view = await render(<SearchScreen />);
    const input = view.getByPlaceholderText('Search across all conversations...');
    fireEvent.changeText(input, 'xyz');

    await waitFor(() => {
      expect(view.getByText('No results found')).toBeTruthy();
    });
  });

  it('calls search endpoint with query', async () => {
    const {api} = require('../../services/api-client');
    api.get.mockResolvedValue(mockResults);

    const view = await render(<SearchScreen />);
    const input = view.getByPlaceholderText('Search across all conversations...');
    fireEvent.changeText(input, 'AI');

    await waitFor(() => {
      expect(api.get).toHaveBeenCalledWith(
        expect.stringContaining('/chat/sessions/search?q=AI'),
      );
    });
  });

  it('renders search results', async () => {
    const {api} = require('../../services/api-client');
    api.get.mockResolvedValue(mockResults);

    const view = await render(<SearchScreen />);
    const input = view.getByPlaceholderText('Search across all conversations...');
    fireEvent.changeText(input, 'AI');

    await waitFor(() => {
      expect(view.getAllByText(/Chat about AI/).length).toBeGreaterThanOrEqual(1);
      expect(view.getByText(/Code review/)).toBeTruthy();
    });
  });

  it('renders all match contents in results', async () => {
    const {api} = require('../../services/api-client');
    api.get.mockResolvedValue(mockResults);

    const view = await render(<SearchScreen />);
    const input = view.getByPlaceholderText('Search across all conversations...');
    fireEvent.changeText(input, 'AI');

    await waitFor(() => {
      expect(view.getByText('What is AI?')).toBeTruthy();
      expect(view.getByText('AI stands for artificial intelligence')).toBeTruthy();
      expect(view.getByText('Review this code')).toBeTruthy();
    });
  });

  it('calls loadSession on result press', async () => {
    const {api} = require('../../services/api-client');
    api.get.mockResolvedValue(mockResults);

    const view = await render(<SearchScreen />);
    const input = view.getByPlaceholderText('Search across all conversations...');
    fireEvent.changeText(input, 'AI');

    await waitFor(() => {
      expect(view.getAllByText(/Chat about AI/).length).toBeGreaterThanOrEqual(1);
    });

    fireEvent.press(view.getAllByText(/Chat about AI/)[0]);
    expect(mockLoadSession).toHaveBeenCalledWith('s1');
  });

  it('handles API error gracefully', async () => {
    const {api} = require('../../services/api-client');
    api.get.mockRejectedValue(new Error('Network error'));

    const view = await render(<SearchScreen />);
    const input = view.getByPlaceholderText('Search across all conversations...');
    fireEvent.changeText(input, 'AI');

    await waitFor(() => {
      expect(view.getByText('No results found')).toBeTruthy();
    });
  });
});
