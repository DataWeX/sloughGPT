import React from 'react';
import {render, waitFor, fireEvent} from '@/test-utils';

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
const {WorkflowScreen} = require('../WorkflowScreen');

describe('WorkflowScreen', () => {
  it('renders title', async () => {
    const {getByText} = await render(<WorkflowScreen />);
    await waitFor(() => {
      expect(getByText('Workflow')).toBeTruthy();
    });
  });

  it('shows pipeline section', async () => {
    const {getByText} = await render(<WorkflowScreen />);
    await waitFor(() => {
      expect(getByText('Pipeline')).toBeTruthy();
    });
  });

  it('shows configuration section', async () => {
    const {getByText} = await render(<WorkflowScreen />);
    await waitFor(() => {
      expect(getByText('Configuration')).toBeTruthy();
    });
  });

  it('shows manual triggers section', async () => {
    const {getByText} = await render(<WorkflowScreen />);
    await waitFor(() => {
      expect(getByText('Manual Triggers')).toBeTruthy();
    });
  });

  it('shows KPI grid with feedback count', async () => {
    const mockApi = require('../../services/api-client').api;
    mockApi.get.mockResolvedValueOnce({
      status: 'running',
      feedback_recorded: 42,
      auto_train_steps: 100,
      workflow_runs: 5,
      last_run: null,
      aggregate_interval: 3600,
      prune_interval: 7200,
      export_interval: 14400,
      health_check_interval: 300,
    });
    const {getByText} = await render(<WorkflowScreen />);
    await waitFor(() => {
      expect(getByText('42')).toBeTruthy();
      expect(getByText('100')).toBeTruthy();
      expect(getByText('5')).toBeTruthy();
    });
  });

  it('shows Start button when stopped', async () => {
    const {getByText} = await render(<WorkflowScreen />);
    await waitFor(() => {
      expect(getByText('Start')).toBeTruthy();
    });
  });

  it('calls start on button press', async () => {
    const mockApi = require('../../services/api-client').api;
    mockApi.get.mockResolvedValueOnce(null);
    mockApi.post.mockResolvedValueOnce({});
    const {getByText} = await render(<WorkflowScreen />);
    await waitFor(() => {
      expect(getByText('Start')).toBeTruthy();
    });
    fireEvent.press(getByText('Start'));
    await waitFor(() => {
      expect(mockApi.post).toHaveBeenCalledWith('/workflow/start');
    });
  });
});
