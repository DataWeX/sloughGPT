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
  useRoute: () => ({params: {datasetId: 'test-ds'}}),
  useNavigation: () => ({goBack: jest.fn()}),
}));

jest.mock('../../services/api-client', () => ({
  api: {
    get: jest.fn().mockImplementation((url: string) => {
      if (url.includes('/stats')) return Promise.resolve({total_rows: 100, avg_length: 50, total_chars: 5000, format: 'jsonl'});
      if (url.includes('/preview')) return Promise.resolve({headers: ['text', 'label'], rows: [['hello', 'greet'], ['world', 'farewell']]});
      return Promise.resolve({id: 'test-ds', name: 'Test Dataset', description: 'A test', row_count: 100, total_chars: 5000, format: 'jsonl', source: 'local', tags: ['test'], created_at: '2025-01-01', updated_at: '2025-01-02'});
    }),
    delete: jest.fn().mockResolvedValue(undefined),
  },
}));

jest.mock('../../services/haptics', () => ({triggerHaptic: jest.fn()}));
jest.mock('../../services/toast', () => ({toast: {success: jest.fn(), error: jest.fn()}}));

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
const {DatasetDetailScreen} = require('../DatasetDetailScreen');

describe('DatasetDetailScreen', () => {
  it('renders dataset name', async () => {
    const {getByText} = await render(<DatasetDetailScreen />);
    await waitFor(() => expect(getByText('Test Dataset')).toBeTruthy());
  });

  it('shows info card', async () => {
    const {getByText} = await render(<DatasetDetailScreen />);
    await waitFor(() => expect(getByText('Dataset Info')).toBeTruthy());
  });

  it('shows stats card', async () => {
    const {getByText} = await render(<DatasetDetailScreen />);
    await waitFor(() => expect(getByText('Statistics')).toBeTruthy());
  });

  it('shows row count', async () => {
    const {getByText} = await render(<DatasetDetailScreen />);
    await waitFor(() => expect(getByText('100')).toBeTruthy());
  });

  it('shows preview card', async () => {
    const {getByText} = await render(<DatasetDetailScreen />);
    await waitFor(() => expect(getByText('Preview')).toBeTruthy());
  });

  it('shows preview rows', async () => {
    const {getByText} = await render(<DatasetDetailScreen />);
    await waitFor(() => expect(getByText('hello | greet')).toBeTruthy());
  });

  it('shows delete button', async () => {
    const {getByText} = await render(<DatasetDetailScreen />);
    await waitFor(() => expect(getByText('Delete Dataset')).toBeTruthy());
  });

  it('shows description', async () => {
    const {getByText} = await render(<DatasetDetailScreen />);
    await waitFor(() => expect(getByText('A test')).toBeTruthy());
  });

  it('shows tags', async () => {
    const {getByText} = await render(<DatasetDetailScreen />);
    await waitFor(() => expect(getByText('test')).toBeTruthy());
  });

  it('shows source', async () => {
    const {getByText} = await render(<DatasetDetailScreen />);
    await waitFor(() => expect(getByText('Source')).toBeTruthy());
  });
});
