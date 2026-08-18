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

jest.mock('@react-navigation/native', () => ({
  useRoute: () => ({params: {modelId: 'gpt2'}}),
  useNavigation: () => ({goBack: jest.fn()}),
}));

jest.mock('../../services/api-client', () => ({
  api: {
    get: jest.fn().mockResolvedValue(null),
    post: jest.fn().mockResolvedValue({coherence: 0.85, repetition: 0.12, perplexity: 42.5, avg_length: 64}),
  },
}));

jest.mock('../../services/haptics', () => ({triggerHaptic: jest.fn()}));
jest.mock('../../services/toast', () => ({toast: {success: jest.fn(), error: jest.fn()}}));

jest.mock('../../stores/model-store', () => ({
  useModelStore: () => ({
    models: [{id: 'gpt2', name: 'GPT-2', type: 'text', params: '124M', source: 'huggingface', loaded: true, tags: ['text-generation']}],
    currentModel: 'gpt2',
    health: {status: 'healthy', model_type: 'gpt2', inference_count: 42},
    loadModel: jest.fn().mockResolvedValue(undefined),
    unloadModel: jest.fn().mockResolvedValue(undefined),
    loadingModelId: null,
  }),
}));

jest.mock('../../components/StatusBadge', () => ({
  StatusBadge: ({label, variant}: any) => {
    const React = require('react');
    const {Text} = require('react-native');
    return React.createElement(Text, {testID: `badge-${variant}`}, label);
  },
}));

jest.mock('../../components/Icon', () => ({
  Icon: ({name}: any) => {
    const React = require('react');
    const {Text} = require('react-native');
    return React.createElement(Text, null, name);
  },
}));

beforeEach(() => jest.clearAllMocks());

// eslint-disable-next-line @typescript-eslint/no-var-requires
const {ModelDetailScreen} = require('../ModelDetailScreen');

describe('ModelDetailScreen', () => {
  it('renders model id in header', async () => {
    const {getByText} = await render(<ModelDetailScreen />);
    await waitFor(() => expect(getByText('gpt2')).toBeTruthy());
  });

  it('shows Loaded badge', async () => {
    const {getByText} = await render(<ModelDetailScreen />);
    await waitFor(() => expect(getByText('Loaded')).toBeTruthy());
  });

  it('shows model info card', async () => {
    const {getByText} = await render(<ModelDetailScreen />);
    await waitFor(() => expect(getByText('Model Info')).toBeTruthy());
  });

  it('shows unload button when loaded', async () => {
    const {getByText} = await render(<ModelDetailScreen />);
    await waitFor(() => expect(getByText('Unload')).toBeTruthy());
  });

  it('shows benchmark button', async () => {
    const {getByText} = await render(<ModelDetailScreen />);
    await waitFor(() => expect(getByText('Benchmark')).toBeTruthy());
  });

  it('shows server health', async () => {
    const {getByText} = await render(<ModelDetailScreen />);
    await waitFor(() => expect(getByText('healthy')).toBeTruthy());
  });

  it('shows inference count', async () => {
    const {getByText} = await render(<ModelDetailScreen />);
    await waitFor(() => expect(getByText('42')).toBeTruthy());
  });
});
