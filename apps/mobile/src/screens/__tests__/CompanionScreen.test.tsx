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
    post: jest.fn().mockResolvedValue({}),
  },
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

beforeEach(() => {
  jest.clearAllMocks();
});

// eslint-disable-next-line @typescript-eslint/no-var-requires
const {CompanionScreen} = require('../CompanionScreen');

describe('CompanionScreen', () => {
  it('renders title', async () => {
    const {getByText} = await render(<CompanionScreen />);
    await waitFor(() => {
      expect(getByText('Companion')).toBeTruthy();
    });
  });

  it('shows personality traits section', async () => {
    const {getByText} = await render(<CompanionScreen />);
    await waitFor(() => {
      expect(getByText('Personality Traits')).toBeTruthy();
    });
  });

  it('shows test chat section', async () => {
    const {getByText} = await render(<CompanionScreen />);
    await waitFor(() => {
      expect(getByText('Test Chat')).toBeTruthy();
    });
  });

  it('shows presets section when data has presets', async () => {
    const mockApi = require('../../services/api-client').api;
    mockApi.get.mockResolvedValueOnce({
      active_preset: 'warm',
      warmth: 0.7,
      curiosity: 0.5,
      playfulness: 0.3,
      confidence: 0.6,
      empathy: 0.8,
      system_prompt: 'You are a companion',
      presets: ['warm', 'professional', 'playful'],
    });
    const {getAllByText} = await render(<CompanionScreen />);
    await waitFor(() => {
      expect(getAllByText('Presets').length).toBeGreaterThanOrEqual(1);
      expect(getAllByText('warm').length).toBeGreaterThanOrEqual(1);
      expect(getAllByText('professional').length).toBeGreaterThanOrEqual(1);
    });
  });

  it('shows save button', async () => {
    const {getByText} = await render(<CompanionScreen />);
    await waitFor(() => {
      expect(getByText('Save Personality')).toBeTruthy();
    });
  });
});
