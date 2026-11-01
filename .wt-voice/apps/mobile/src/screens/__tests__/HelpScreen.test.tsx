import React from 'react';
import {render, fireEvent} from '@/test-utils';
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

const {HelpScreen} = require('../HelpScreen');

describe('HelpScreen', () => {
  it('renders title', async () => {
    const {getByText} = await render(<HelpScreen />);
    expect(getByText('Help')).toBeTruthy();
  });

  it('renders all FAQ questions', async () => {
    const {getByText} = await render(<HelpScreen />);
    expect(getByText('How do I start chatting?')).toBeTruthy();
    expect(getByText('How do I train my own model?')).toBeTruthy();
    expect(getByText('What models work best?')).toBeTruthy();
    expect(getByText('What is a "soul"?')).toBeTruthy();
    expect(getByText('How do I add knowledge?')).toBeTruthy();
    expect(getByText('What training text works?')).toBeTruthy();
    expect(getByText('How do I export data?')).toBeTruthy();
    expect(getByText('Can I use this offline?')).toBeTruthy();
  });

  it('renders FAQ plus arrows', async () => {
    const {getAllByText} = await render(<HelpScreen />);
    const pluses = getAllByText('+');
    expect(pluses.length).toBe(8);
  });

  it('renders quick start steps', async () => {
    const {getByText} = await render(<HelpScreen />);
    expect(getByText('Quick Start')).toBeTruthy();
    expect(getByText(/Connect to your server/)).toBeTruthy();
  });

  it('renders keyboard shortcuts', async () => {
    const {getByText} = await render(<HelpScreen />);
    expect(getByText('Keyboard Shortcuts')).toBeTruthy();
    expect(getByText('Send message')).toBeTruthy();
  });

  it('renders troubleshooting card', async () => {
    const {getByText} = await render(<HelpScreen />);
    expect(getByText('Troubleshooting')).toBeTruthy();
    expect(getByText(/Connection refused/)).toBeTruthy();
  });
});
