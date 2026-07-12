/**
 * Tests for BookmarksScreen.
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

jest.mock('../../services/bookmarks', () => ({
  getBookmarks: jest.fn(),
  removeBookmark: jest.fn(),
}));

jest.mock('../../services/haptics', () => ({
  triggerHaptic: jest.fn(),
}));

// eslint-disable-next-line @typescript-eslint/no-var-requires
const {BookmarksScreen} = require('../BookmarksScreen');

const mockBookmarks = [
  {id: 'b1', sessionId: 's1', content: 'Hello world', role: 'user', savedAt: Date.now() - 1000},
  {id: 'b2', sessionId: 's2', content: 'AI response text', role: 'assistant', savedAt: Date.now()},
];

describe('BookmarksScreen', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    const {getBookmarks} = require('../../services/bookmarks');
    getBookmarks.mockResolvedValue(mockBookmarks);
  });

  it('renders without crashing', async () => {
    const view = await render(<BookmarksScreen />);
    expect(view.getByText('Bookmarks')).toBeTruthy();
  });

  it('shows bookmark count', async () => {
    const view = await render(<BookmarksScreen />);
    await waitFor(() => {
      expect(view.getByText('2 saved messages')).toBeTruthy();
    });
  });

  it('renders bookmark items', async () => {
    const view = await render(<BookmarksScreen />);
    await waitFor(() => {
      expect(view.getByText('Hello world')).toBeTruthy();
      expect(view.getByText('AI response text')).toBeTruthy();
    });
  });

  it('shows empty state when no bookmarks', async () => {
    const {getBookmarks} = require('../../services/bookmarks');
    getBookmarks.mockResolvedValue([]);
    const view = await render(<BookmarksScreen />);
    await waitFor(() => {
      expect(view.getByText('No bookmarks yet')).toBeTruthy();
    });
  });

  it('calls removeBookmark on long press and confirm', async () => {
    const {removeBookmark} = require('../../services/bookmarks');
    const {Alert} = require('react-native');
    const alertMock = jest.fn();
    Alert.alert = alertMock;

    const view = await render(<BookmarksScreen />);

    await waitFor(() => {
      expect(view.getByText('Hello world')).toBeTruthy();
    });

    const items = view.getAllByText(/Hello world|AI response text/);
    fireEvent(items[0], 'longPress');

    const removeFn = alertMock.mock.calls[0][2][1].onPress;
    await removeFn();

    expect(removeBookmark).toHaveBeenCalledWith('b1');
  });

  it('calls loadSession on tap', async () => {
    const view = await render(<BookmarksScreen />);

    await waitFor(() => {
      expect(view.getByText('Hello world')).toBeTruthy();
    });

    fireEvent.press(view.getByText('Hello world'));
    expect(mockLoadSession).toHaveBeenCalledWith('s1');
  });

  it('shows role labels correctly', async () => {
    const view = await render(<BookmarksScreen />);
    await waitFor(() => {
      expect(view.getByText('You')).toBeTruthy();
      expect(view.getByText('AI')).toBeTruthy();
    });
  });
});
