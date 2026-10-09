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

jest.mock('react-native/Libraries/Linking/Linking', () => ({
  openURL: jest.fn().mockResolvedValue(undefined),
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
const {LegalScreen} = require('../LegalScreen');

describe('LegalScreen', () => {
  it('renders title', async () => {
    const {getByText} = await render(<LegalScreen />);
    await waitFor(() => {
      expect(getByText('Legal')).toBeTruthy();
    });
  });

  it('shows privacy policy entry', async () => {
    const {getByText} = await render(<LegalScreen />);
    await waitFor(() => {
      expect(getByText('Privacy Policy')).toBeTruthy();
    });
  });

  it('shows terms of service entry', async () => {
    const {getByText} = await render(<LegalScreen />);
    await waitFor(() => {
      expect(getByText('Terms of Service')).toBeTruthy();
    });
  });

  it('shows open source licenses entry', async () => {
    const {getByText} = await render(<LegalScreen />);
    await waitFor(() => {
      expect(getByText('Open Source Licenses')).toBeTruthy();
    });
  });

  it('shows website entry', async () => {
    const {getByText} = await render(<LegalScreen />);
    await waitFor(() => {
      expect(getByText('Visit Website')).toBeTruthy();
    });
  });

  it('shows source code entry', async () => {
    const {getByText} = await render(<LegalScreen />);
    await waitFor(() => {
      expect(getByText('Source Code')).toBeTruthy();
    });
  });

  it('expands privacy policy on press', async () => {
    const {getByText} = await render(<LegalScreen />);
    await waitFor(() => {
      expect(getByText('Privacy Policy')).toBeTruthy();
    });
  });
});
