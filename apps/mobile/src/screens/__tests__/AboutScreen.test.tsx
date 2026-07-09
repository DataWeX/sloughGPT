import React from 'react';
import {render} from '@/test-utils';
import {View, Text} from 'react-native';

jest.mock('react-native-safe-area-context', () => {
  const React = require('react');
  const {View} = require('react-native');
  return {
    SafeAreaView: ({children, edges, style}: any) =>
      React.createElement(View, {style, testID: 'safe-area-view'}, children),
    useSafeAreaInsets: () => ({top: 0, bottom: 0, left: 0, right: 0}),
  };
});

jest.mock('../../components/Icon', () => ({
  Icon: ({name, size, color}: any) => {
    const React = require('react');
    const {Text} = require('react-native');
    return React.createElement(Text, {testID: `icon-${name}`}, `[${name}]`);
  },
}));

const {AboutScreen} = require('../AboutScreen');

describe('AboutScreen', () => {
  it('renders app name and version', async () => {
    const {getByText} = await render(<AboutScreen />);
    expect(getByText('SloughGPT')).toBeTruthy();
    expect(getByText('v1.0.0')).toBeTruthy();
  });

  it('renders about card', async () => {
    const {getAllByText, getByText} = await render(<AboutScreen />);
    expect(getAllByText('About').length).toBeGreaterThanOrEqual(1);
    expect(getByText(/AI platform/)).toBeTruthy();
  });

  it('renders features card', async () => {
    const {getByText} = await render(<AboutScreen />);
    expect(getByText('Features')).toBeTruthy();
    expect(getByText(/Real-time chat/)).toBeTruthy();
    expect(getByText(/Model management/)).toBeTruthy();
  });

  it('renders architecture card', async () => {
    const {getByText} = await render(<AboutScreen />);
    expect(getByText('Architecture')).toBeTruthy();
    expect(getByText(/React Native CLI/)).toBeTruthy();
  });

  it('renders keyboard shortcuts card', async () => {
    const {getByText} = await render(<AboutScreen />);
    expect(getByText('Keyboard Shortcuts')).toBeTruthy();
    expect(getByText('Enter — Send message')).toBeTruthy();
  });
});
