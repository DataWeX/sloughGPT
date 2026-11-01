import React from 'react';
import {render} from '@/test-utils';

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

jest.mock('../../components/StatusBadge', () => ({
  StatusBadge: ({label, variant}: any) => {
    const React = require('react');
    const {Text} = require('react-native');
    return React.createElement(Text, {testID: `status-${variant}`}, label);
  },
}));

jest.mock('../../services/api-client', () => ({
  api: {
    get: jest.fn().mockResolvedValue({
      model_loaded: true,
      model_name: 'test-model',
      uptime_s: 3600,
      request_count: 42,
      error_count: 1,
    }),
  },
}));

jest.mock('../../stores/model-store', () => ({
  useModelStore: () => ({
    health: {model_loaded: true, model_name: 'test-model'},
    models: [{id: 'm1', name: 'Model 1'}],
    currentSoul: {name: 'Friendly', description: 'A friendly soul'},
    refresh: jest.fn().mockResolvedValue(undefined),
  }),
}));

jest.mock('../../stores/chat-store', () => ({
  useChatStore: () => ({
    sessions: [
      {id: 's1', name: 'Test chat', message_count: 5},
      {id: 's2', name: 'Another chat', message_count: 3},
    ],
    refreshSessions: jest.fn().mockResolvedValue(undefined),
    loadSession: jest.fn(),
  }),
}));

jest.mock('../../contexts/SidebarContext', () => ({
  SidebarProvider: ({children}: any) => children,
  useSidebar: () => ({
    open: jest.fn(),
    navigate: jest.fn(),
    activeScreen: 'Home',
  }),
}));

const {HomeScreen} = require('../HomeScreen');

describe('HomeScreen', () => {
  it('renders header', async () => {
    const {getByText} = await render(<HomeScreen />);
    expect(getByText('Home')).toBeTruthy();
    expect(getByText('SloughGPT Mobile')).toBeTruthy();
  });

  it('shows model status', async () => {
    const {getByText} = await render(<HomeScreen />);
    expect(getByText('Model')).toBeTruthy();
    expect(getByText('test-model')).toBeTruthy();
  });

  it('shows soul status', async () => {
    const {getByText} = await render(<HomeScreen />);
    expect(getByText('Soul')).toBeTruthy();
    expect(getByText('Friendly')).toBeTruthy();
    expect(getByText('A friendly soul')).toBeTruthy();
  });

  it('renders quick actions', async () => {
    const {getByText, getAllByText} = await render(<HomeScreen />);
    expect(getByText('Quick Actions')).toBeTruthy();
    expect(getByText('Chat')).toBeTruthy();
    expect(getAllByText('Models').length).toBeGreaterThanOrEqual(1);
    expect(getByText('Train')).toBeTruthy();
    expect(getByText('Datasets')).toBeTruthy();
  });

  it('renders system stats', async () => {
    const {getByText} = await render(<HomeScreen />);
    expect(getByText('System')).toBeTruthy();
    expect(getByText('Uptime')).toBeTruthy();
    expect(getByText('1h 0m')).toBeTruthy();
    expect(getByText('Requests')).toBeTruthy();
    expect(getByText('42')).toBeTruthy();
  });

  it('renders recent sessions', async () => {
    const {getByText} = await render(<HomeScreen />);
    expect(getByText('Recent Chats')).toBeTruthy();
    expect(getByText('Test chat')).toBeTruthy();
    expect(getByText('Another chat')).toBeTruthy();
    expect(getByText('5 messages')).toBeTruthy();
    expect(getByText('3 messages')).toBeTruthy();
  });

  it('shows online status when API is reachable', async () => {
    const {getByText} = await render(<HomeScreen />);
    expect(getByText('Online')).toBeTruthy();
  });
});
