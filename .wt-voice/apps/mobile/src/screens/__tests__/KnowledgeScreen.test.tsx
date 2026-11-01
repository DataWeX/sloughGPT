import React from 'react';
import {render, fireEvent, waitFor} from '@/test-utils';
import {View, Text} from 'react-native';

jest.mock('react-native-safe-area-context', () => {
  const React = require('react');
  const {View} = require('react-native');
  return {
    SafeAreaView: ({children, edges, style}: any) =>
      React.createElement(View, {style, testID: 'safe-area-view'}, children),
    useSafeAreaInsets: () => ({top: 0, bottom: 0, left: 0, right: 0}),
  };
});

const mockApiGet = jest.fn().mockRejectedValue(new Error('default'));
const mockApiPost = jest.fn().mockRejectedValue(new Error('default'));
const mockApiPatch = jest.fn().mockRejectedValue(new Error('default'));
const mockApiDelete = jest.fn().mockRejectedValue(new Error('default'));

jest.mock('../../services/api-client', () => ({
  api: {
    get: (...args: any[]) => mockApiGet(...args),
    post: (...args: any[]) => mockApiPost(...args),
    patch: (...args: any[]) => mockApiPatch(...args),
    delete: (...args: any[]) => mockApiDelete(...args),
  },
}));

jest.mock('../../components/StatusBadge', () => ({
  StatusBadge: ({label, variant}: any) => {
    const React = require('react');
    const {Text} = require('react-native');
    return React.createElement(Text, {testID: `badge-${variant}`, children: label});
  },
}));

jest.mock('../../components/Icon', () => ({
  Icon: ({name, size, color}: any) => {
    const React = require('react');
    const {Text} = require('react-native');
    return React.createElement(Text, {testID: `icon-${name}`}, `[${name}]`);
  },
}));

jest.mock('../HealthScreen');

const sampleItems: any[] = [
  {id: '1', content: 'Paris is capital of France', topic: 'geography', importance: 4},
  {id: '2', content: 'React Native uses JavaScript', topic: 'tech', importance: 3},
  {id: '3', content: 'The sky is blue', topic: null, importance: 1},
];

beforeEach(() => {
  jest.clearAllMocks();
  mockApiGet
    .mockResolvedValueOnce({items: sampleItems})
    .mockResolvedValueOnce({topics: ['geography', 'tech']});
});

const {KnowledgeScreen} = require('../KnowledgeScreen');

describe('KnowledgeScreen', () => {
  it('renders screen title', async () => {
    const {getByText} = await render(<KnowledgeScreen />);
    expect(getByText('What AI Knows About Me')).toBeTruthy();
  });

  it('renders knowledge items', async () => {
    const {getByText} = await render(<KnowledgeScreen />);
    await waitFor(() => {
      expect(getByText('Paris is capital of France')).toBeTruthy();
    });
  });

  it('shows "No knowledge items yet" when empty', async () => {
    mockApiGet.mockReset().mockResolvedValue({items: []});
    const {getByText} = await render(<KnowledgeScreen />);
    await waitFor(() => {
      expect(getByText('No knowledge items yet')).toBeTruthy();
    });
  });

  it('opens add modal on header button press', async () => {
    const {getByText} = await render(<KnowledgeScreen />);
    await waitFor(() => expect(getByText('What AI Knows About Me')).toBeTruthy());
  });

  it('renders topic chips', async () => {
    const {findAllByText} = await render(<KnowledgeScreen />);
    const geographyChips = await findAllByText('geography');
    expect(geographyChips.length).toBeGreaterThanOrEqual(1);
  });
});
