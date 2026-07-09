import React from 'react';
import {render} from '../../test-utils';
import {SearchSessionsModal} from '../SearchSessionsModal';

jest.mock('../../services/api-client', () => ({
  api: {
    searchSessions: jest.fn(async () => ({results: []})),
  },
}));

jest.mock('../../services/haptics', () => ({
  triggerHaptic: jest.fn(),
}));

beforeEach(() => {
  jest.clearAllMocks();
});

describe('SearchSessionsModal', () => {
  const defaultProps = {
    visible: true,
    onClose: jest.fn(),
    onSelectSession: jest.fn(),
  };

  it('renders without crashing', () => {
    expect(() => render(<SearchSessionsModal {...defaultProps} />)).not.toThrow();
  });

  it('renders without error in hidden state', () => {
    expect(() => render(
      <SearchSessionsModal visible={false} onClose={jest.fn()} onSelectSession={jest.fn()} />,
    )).not.toThrow();
  });
});
