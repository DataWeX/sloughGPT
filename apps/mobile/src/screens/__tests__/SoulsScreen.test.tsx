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
    post: jest.fn().mockResolvedValue(null),
  },
}));

const mockRefresh = jest.fn().mockResolvedValue(undefined);
const mockSwitchSoul = jest.fn().mockResolvedValue(undefined);

jest.mock('../../stores/model-store', () => ({
  useModelStore: jest.fn(() => ({
    souls: [
      {name: 'friendly', description: 'A friendly assistant', traits: ['warm', 'helpful']},
      {name: 'formal', description: 'A formal assistant', traits: ['professional', 'precise']},
    ],
    currentSoul: {name: 'friendly', description: 'A friendly assistant', traits: ['warm', 'helpful']},
    checkpoints: [],
    health: {model_name: 'gpt2'},
    refresh: mockRefresh,
    switchSoul: mockSwitchSoul,
  })),
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

import {api} from '../../services/api-client';
import {useModelStore} from '../../stores/model-store';
const mockApiGet = api.get as jest.Mock;

beforeEach(() => {
  jest.clearAllMocks();
});

// eslint-disable-next-line @typescript-eslint/no-var-requires
const {SoulsScreen} = require('../SoulsScreen');

describe('SoulsScreen', () => {
  it('renders title', async () => {
    mockApiGet.mockResolvedValue({name: 'friendly', traits: [], personality: {}});
    const {getByText} = await render(<SoulsScreen />);
    await waitFor(() => {
      expect(getByText('Souls')).toBeTruthy();
    });
  });

  it('lists available souls', async () => {
    mockApiGet.mockResolvedValue({name: 'friendly', traits: [], personality: {}});
    const {getByText} = await render(<SoulsScreen />);
    await waitFor(() => {
      expect(getByText('friendly')).toBeTruthy();
      expect(getByText('formal')).toBeTruthy();
    });
  });

  it('shows Active badge for current soul', async () => {
    mockApiGet.mockResolvedValue({name: 'friendly', traits: [], personality: {}});
    const {getByText} = await render(<SoulsScreen />);
    await waitFor(() => {
      expect(getByText('Active')).toBeTruthy();
    });
  });

  it('shows soul descriptions', async () => {
    mockApiGet.mockResolvedValue({name: 'friendly', traits: [], personality: {}});
    const {getByText} = await render(<SoulsScreen />);
    await waitFor(() => {
      expect(getByText('A friendly assistant')).toBeTruthy();
      expect(getByText('A formal assistant')).toBeTruthy();
    });
  });
});
