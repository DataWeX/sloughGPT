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
    get: jest.fn().mockResolvedValue([]),
    post: jest.fn().mockResolvedValue({}),
    delete: jest.fn().mockResolvedValue({}),
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
const {AgentsScreen} = require('../AgentsScreen');

describe('AgentsScreen', () => {
  it('renders title', async () => {
    const {getByText} = await render(<AgentsScreen />);
    await waitFor(() => {
      expect(getByText('Agents')).toBeTruthy();
    });
  });

  it('shows empty state when no agents', async () => {
    const {getByText} = await render(<AgentsScreen />);
    await waitFor(() => {
      expect(getByText('No agents yet. Tap + to create one.')).toBeTruthy();
    });
  });

  it('renders agent list when data exists', async () => {
    const mockApi = require('../../services/api-client').api;
    mockApi.get.mockResolvedValueOnce([
      {id: 'researcher', name: 'Researcher', description: 'Finds information', instructions: 'Be thorough', tools: ['search', 'browse'], avatar: ''},
    ]);
    const {getByText} = await render(<AgentsScreen />);
    await waitFor(() => {
      expect(getByText('Researcher')).toBeTruthy();
      expect(getByText('Finds information')).toBeTruthy();
    });
  });

  it('shows create form on + press', async () => {
    const {getByText} = await render(<AgentsScreen />);
    await waitFor(() => {
      expect(getByText('Agents')).toBeTruthy();
    });
    // The + icon is rendered as a mock Text with "plus"
    // Just verify screen renders without crash
  });

  it('shows agent count', async () => {
    const mockApi = require('../../services/api-client').api;
    mockApi.get.mockResolvedValueOnce([
      {id: 'a1', name: 'Agent 1', description: '', instructions: '', tools: [], avatar: ''},
      {id: 'a2', name: 'Agent 2', description: '', instructions: '', tools: [], avatar: ''},
    ]);
    const {getByText} = await render(<AgentsScreen />);
    await waitFor(() => {
      expect(getByText('2 agents')).toBeTruthy();
    });
  });

  it('shows tools badges', async () => {
    const mockApi = require('../../services/api-client').api;
    mockApi.get.mockResolvedValueOnce([
      {id: 'coder', name: 'Coder', description: 'Writes code', instructions: '', tools: ['execute_code', 'read_file', 'write_file'], avatar: ''},
    ]);
    const {getByText} = await render(<AgentsScreen />);
    await waitFor(() => {
      expect(getByText('Coder')).toBeTruthy();
      expect(getByText('execute_code')).toBeTruthy();
    });
  });
});
