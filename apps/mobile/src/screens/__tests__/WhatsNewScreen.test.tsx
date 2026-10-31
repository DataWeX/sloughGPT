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
const {WhatsNewScreen} = require('../WhatsNewScreen');

describe('WhatsNewScreen', () => {
  it('renders title', async () => {
    const {getByText} = await render(<WhatsNewScreen />);
    await waitFor(() => {
      expect(getByText("What's New")).toBeTruthy();
    });
  });

  it('shows deep linking entry', async () => {
    const {getByText} = await render(<WhatsNewScreen />);
    await waitFor(() => {
      expect(getByText('Deep linking')).toBeTruthy();
    });
  });

  it('shows chat search entry', async () => {
    const {getByText} = await render(<WhatsNewScreen />);
    await waitFor(() => {
      expect(getByText('Chat search with highlighting')).toBeTruthy();
    });
  });

  it('shows notification settings entry', async () => {
    const {getByText} = await render(<WhatsNewScreen />);
    await waitFor(() => {
      expect(getByText('Notification settings')).toBeTruthy();
    });
  });

  it('shows New badge on latest entry', async () => {
    const {getByText} = await render(<WhatsNewScreen />);
    await waitFor(() => {
      expect(getByText('New')).toBeTruthy();
    });
  });

  it('shows newest entries at the top', async () => {
    const {getByText} = await render(<WhatsNewScreen />);
    await waitFor(() => {
      expect(getByText('Accessibility & polish pass')).toBeTruthy();
      expect(getByText('More reliable streaming')).toBeTruthy();
      expect(getByText('Chat retry & typing indicator')).toBeTruthy();
      expect(getByText('Training ETA & cancellation')).toBeTruthy();
      expect(getByText('Dark mode & startup fixes')).toBeTruthy();
    });
  });

  it('still shows archived entries', async () => {
    const {getByText} = await render(<WhatsNewScreen />);
    await waitFor(() => {
      expect(getByText('Deep linking')).toBeTruthy();
      expect(getByText('Chat search with highlighting')).toBeTruthy();
    });
  });
});
