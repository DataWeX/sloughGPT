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
    delete: jest.fn().mockResolvedValue(null),
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
const {FilesScreen} = require('../FilesScreen');

describe('FilesScreen', () => {
  it('renders title', async () => {
    const {getByText} = await render(<FilesScreen />);
    await waitFor(() => {
      expect(getByText('Files')).toBeTruthy();
    });
  });

  it('shows empty state when no files', async () => {
    const {getByText} = await render(<FilesScreen />);
    await waitFor(() => {
      expect(getByText('No files')).toBeTruthy();
    });
  });

  it('shows file count', async () => {
    const mockApi = require('../../services/api-client').api;
    mockApi.get.mockResolvedValueOnce([
      {id: '1', filename: 'test.txt', size: 1024, content_type: 'txt', uploaded_at: new Date().toISOString(), ingested: true},
      {id: '2', filename: 'data.csv', size: 2048, content_type: 'csv', uploaded_at: new Date().toISOString(), ingested: false},
    ]);
    const {getByText} = await render(<FilesScreen />);
    await waitFor(() => {
      expect(getByText('2 files')).toBeTruthy();
    });
  });

  it('displays files with correct names', async () => {
    const mockApi = require('../../services/api-client').api;
    mockApi.get.mockResolvedValueOnce([
      {id: '1', filename: 'readme.md', size: 512, content_type: 'md', uploaded_at: new Date().toISOString(), ingested: true, chunk_count: 5},
    ]);
    const {getByText} = await render(<FilesScreen />);
    await waitFor(() => {
      expect(getByText('readme.md')).toBeTruthy();
      expect(getByText('Ingested')).toBeTruthy();
      expect(getByText('5 chunks')).toBeTruthy();
    });
  });

  it('shows search input', async () => {
    const {getByPlaceholderText} = await render(<FilesScreen />);
    await waitFor(() => {
      expect(getByPlaceholderText('Search files...')).toBeTruthy();
    });
  });
});
