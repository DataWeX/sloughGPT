import React from 'react';
import {render, waitFor} from '@/test-utils';

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
const {SplashScreen} = require('../SplashAnimated');

describe('SplashScreen', () => {
  it('renders SG logo', async () => {
    const {getByText} = await render(<SplashScreen />);
    await waitFor(() => {
      expect(getByText('SG')).toBeTruthy();
    });
  });

  it('renders app name', async () => {
    const {getByText} = await render(<SplashScreen />);
    await waitFor(() => {
      expect(getByText('SloughGPT')).toBeTruthy();
    });
  });

  it('renders tagline', async () => {
    const {getByText} = await render(<SplashScreen />);
    await waitFor(() => {
      expect(getByText('Self-hosted AI assistant')).toBeTruthy();
    });
  });
});
