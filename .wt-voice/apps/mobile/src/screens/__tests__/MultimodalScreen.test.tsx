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
const {MultimodalScreen} = require('../MultimodalScreen');

describe('MultimodalScreen', () => {
  it('renders title', async () => {
    const {getByText} = await render(<MultimodalScreen />);
    await waitFor(() => {
      expect(getByText('Multimodal')).toBeTruthy();
    });
  });

  it('shows status tab by default', async () => {
    const {getAllByText} = await render(<MultimodalScreen />);
    await waitFor(() => {
      expect(getAllByText('Status').length).toBeGreaterThanOrEqual(1);
    });
  });

  it('shows all four tabs', async () => {
    const {getAllByText} = await render(<MultimodalScreen />);
    await waitFor(() => {
      expect(getAllByText('Status').length).toBeGreaterThanOrEqual(1);
      expect(getAllByText('Vision').length).toBeGreaterThanOrEqual(1);
      expect(getAllByText('Audio').length).toBeGreaterThanOrEqual(1);
      expect(getAllByText('Generate').length).toBeGreaterThanOrEqual(1);
    });
  });

  it('shows vision engine section', async () => {
    const {getByText} = await render(<MultimodalScreen />);
    await waitFor(() => {
      expect(getByText('Vision Engine')).toBeTruthy();
    });
  });

  it('shows DPO training section', async () => {
    const {getByText} = await render(<MultimodalScreen />);
    await waitFor(() => {
      expect(getByText('DPO Training')).toBeTruthy();
    });
  });

  it('shows video training section', async () => {
    const {getByText} = await render(<MultimodalScreen />);
    await waitFor(() => {
      expect(getByText('Video Training')).toBeTruthy();
    });
  });

  it('renders with status data', async () => {
    const mockApi = require('../../services/api-client').api;
    mockApi.get.mockResolvedValueOnce({
      engine: {trained: true, vocab_size: 128, learning_count: 50, unique_captions: 30},
      vision: {available: true},
      audio: {available: true, tts_calls: 10},
      dpo: {running: false, status: null},
      video: {training: false, progress: 0},
    });
    const {getByText} = await render(<MultimodalScreen />);
    await waitFor(() => {
      expect(getByText('Trained')).toBeTruthy();
      expect(getByText('128')).toBeTruthy();
      expect(getByText('50')).toBeTruthy();
    });
  });
});
