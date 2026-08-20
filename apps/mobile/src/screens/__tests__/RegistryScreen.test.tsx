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
  },
}));

jest.mock('../../services/haptics', () => ({
  triggerHaptic: jest.fn(),
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
const {RegistryScreen} = require('../RegistryScreen');

describe('RegistryScreen', () => {
  it('renders title', async () => {
    const {getByText} = await render(<RegistryScreen />);
    await waitFor(() => {
      expect(getByText('Registry')).toBeTruthy();
    });
  });

  it('shows empty state when no models', async () => {
    const {getByText} = await render(<RegistryScreen />);
    await waitFor(() => {
      expect(getByText('No registered models')).toBeTruthy();
    });
  });

  it('loads models from API', async () => {
    const mockApi = require('../../services/api-client').api;
    mockApi.get
      .mockResolvedValueOnce({
        models: [
          {model_id: 'gpt2', status: 'loaded', registered_at: new Date().toISOString()},
          {model_id: 'lstm-s1', status: 'healthy', registered_at: new Date().toISOString()},
        ],
      })
      .mockResolvedValueOnce({models_registered: 2, models_loaded: 2, healthy: true});
    const {getByText} = await render(<RegistryScreen />);
    await waitFor(() => {
      expect(getByText('gpt2')).toBeTruthy();
      expect(getByText('lstm-s1')).toBeTruthy();
    });
  });

  it('shows stats cards when stats loaded', async () => {
    const mockApi = require('../../services/api-client').api;
    mockApi.get
      .mockResolvedValueOnce({models: []})
      .mockResolvedValueOnce({models_registered: 5, models_loaded: 3, has_errors: true, degraded: false});
    const {getByText} = await render(<RegistryScreen />);
    await waitFor(() => {
      expect(getByText('5')).toBeTruthy();
      expect(getByText('3')).toBeTruthy();
      expect(getByText('1')).toBeTruthy();
    });
  });

  it('shows model count in header', async () => {
    const mockApi = require('../../services/api-client').api;
    mockApi.get
      .mockResolvedValueOnce({models: [{model_id: 'a', status: 'loaded'}]})
      .mockResolvedValueOnce({models_registered: 1, models_loaded: 1});
    const {getByText} = await render(<RegistryScreen />);
    await waitFor(() => {
      expect(getByText('1/1 loaded')).toBeTruthy();
    });
  });
});
