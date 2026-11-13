import React from 'react';
import {render, fireEvent, waitFor} from '@/test-utils';

const mockNavigate = jest.fn();

jest.mock('react-native-safe-area-context', () => {
  const React = require('react');
  const {View} = require('react-native');
  return {
    SafeAreaView: ({children, style}: any) =>
      React.createElement(View, {style}, children),
    useSafeAreaInsets: () => ({top: 0, bottom: 0, left: 0, right: 0}),
  };
});

jest.mock('../../services/api-client', () => ({
  api: {
    get: jest.fn().mockRejectedValue(new Error('default mock')),
  },
}));

jest.mock('../../components/StatusBadge', () => ({
  StatusBadge: ({label, variant}: any) => {
    const React = require('react');
    const {Text} = require('react-native');
    return React.createElement(Text, {testID: `badge-${variant}`}, label);
  },
}));

jest.mock('../../components/Icon', () => ({
  Icon: ({name, size, color}: any) => {
    const React = require('react');
    const {Text} = require('react-native');
    return React.createElement(Text, {testID: `icon-${name}`}, `${name}:${size}`);
  },
}));

jest.mock('../../hooks/useHapticPress', () => ({
  useHapticPress: () => (_type: string, fn: () => void) => fn(),
}));

jest.mock('@react-navigation/native', () => ({
  useNavigation: () => ({navigate: mockNavigate}),
}));

import {api} from '../../services/api-client';
const mockApiGet = api.get as jest.Mock;

const HEALTH_OK = {
  status: 'healthy',
  model_loaded: true,
  model_name: 'gpt2',
  uptime: 3600,
  inference_count: 42,
};

beforeEach(() => {
  jest.clearAllMocks();
  jest.useFakeTimers();
});

afterEach(() => {
  jest.runOnlyPendingTimers();
  jest.useRealTimers();
});

// eslint-disable-next-line @typescript-eslint/no-var-requires
const {ToolsScreen} = require('../ToolsScreen');

describe('ToolsScreen', () => {
  it('renders title', async () => {
    mockApiGet.mockResolvedValue(HEALTH_OK);
    const {getByText} = await render(<ToolsScreen />);
    expect(getByText('Tools')).toBeTruthy();
  });

  it('shows Connected badge when server healthy', async () => {
    mockApiGet.mockResolvedValue(HEALTH_OK);
    const {getByText} = await render(<ToolsScreen />);
    await waitFor(() => {
      expect(getByText('Connected')).toBeTruthy();
    });
  });

  it('shows Offline badge on API failure', async () => {
    mockApiGet.mockRejectedValue(new Error('network'));
    const {getByText} = await render(<ToolsScreen />);
    await waitFor(() => {
      expect(getByText('Offline')).toBeTruthy();
    });
  });

  it('shows model name when connected', async () => {
    mockApiGet.mockResolvedValue(HEALTH_OK);
    const {getByText} = await render(<ToolsScreen />);
    await waitFor(() => {
      expect(getByText('gpt2')).toBeTruthy();
    });
  });

  it('shows uptime formatted', async () => {
    mockApiGet.mockResolvedValue(HEALTH_OK);
    const {getByText} = await render(<ToolsScreen />);
    await waitFor(() => {
      expect(getByText('1h 0m')).toBeTruthy();
    });
  });

  it('shows inference count', async () => {
    mockApiGet.mockResolvedValue(HEALTH_OK);
    const {getByText} = await render(<ToolsScreen />);
    await waitFor(() => {
      expect(getByText('42')).toBeTruthy();
    });
  });

  it('shows "None loaded" when no model', async () => {
    mockApiGet.mockResolvedValue({...HEALTH_OK, model_name: null});
    const {getByText} = await render(<ToolsScreen />);
    await waitFor(() => {
      expect(getByText('None loaded')).toBeTruthy();
    });
  });

  it('shows "Could not reach server" on failure', async () => {
    mockApiGet.mockRejectedValue(new Error('net'));
    const {getByText} = await render(<ToolsScreen />);
    await waitFor(() => {
      expect(getByText('Could not reach server')).toBeTruthy();
    });
  });

  it('renders all 4 tool links', async () => {
    mockApiGet.mockResolvedValue(HEALTH_OK);
    const {getByText} = await render(<ToolsScreen />);
    await waitFor(() => {
      expect(getByText('Training')).toBeTruthy();
      expect(getByText('Knowledge')).toBeTruthy();
      expect(getByText('Bookmarks')).toBeTruthy();
      expect(getByText('Search')).toBeTruthy();
    });
  });

  it('navigates to Training on press', async () => {
    mockApiGet.mockResolvedValue(HEALTH_OK);
    const {getByText} = await render(<ToolsScreen />);
    await waitFor(() => {
      expect(getByText('Training')).toBeTruthy();
    });
    fireEvent.press(getByText('Training'));
    expect(mockNavigate).toHaveBeenCalledWith('Training');
  });

  it('navigates to Knowledge on press', async () => {
    mockApiGet.mockResolvedValue(HEALTH_OK);
    const {getByText} = await render(<ToolsScreen />);
    await waitFor(() => {
      expect(getByText('Knowledge')).toBeTruthy();
    });
    fireEvent.press(getByText('Knowledge'));
    expect(mockNavigate).toHaveBeenCalledWith('Knowledge');
  });

  it('navigates to Health via View Details', async () => {
    mockApiGet.mockResolvedValue(HEALTH_OK);
    const {getByText} = await render(<ToolsScreen />);
    await waitFor(() => {
      expect(getByText('View Details')).toBeTruthy();
    });
    fireEvent.press(getByText('View Details'));
    expect(mockNavigate).toHaveBeenCalledWith('Health');
  });
});
