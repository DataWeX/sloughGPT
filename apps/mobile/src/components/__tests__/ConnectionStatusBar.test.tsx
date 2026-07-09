import React from 'react';
import {render} from '../../test-utils';
import {ConnectionStatusBar} from '../ConnectionStatusBar';

jest.mock('../../hooks/useConnectionStatus', () => ({
  useConnectionStatus: jest.fn(() => ({
    state: 'connected',
    latencyMs: 42,
    retryCount: 0,
  })),
}));

describe('ConnectionStatusBar', () => {
  it('renders in connected state', () => {
    expect(() => render(<ConnectionStatusBar />)).not.toThrow();
  });

  it('renders with retry callback', () => {
    expect(() => render(<ConnectionStatusBar onRetry={jest.fn()} />)).not.toThrow();
  });

  it('renders in offline state', () => {
    const mock = require('../../hooks/useConnectionStatus');
    mock.useConnectionStatus.mockReturnValueOnce({state: 'offline', latencyMs: null, retryCount: 3});
    expect(() => render(<ConnectionStatusBar onRetry={jest.fn()} />)).not.toThrow();
  });

  it('renders in connecting state', () => {
    const mock = require('../../hooks/useConnectionStatus');
    mock.useConnectionStatus.mockReturnValueOnce({state: 'connecting', latencyMs: null, retryCount: 0});
    expect(() => render(<ConnectionStatusBar />)).not.toThrow();
  });

  it('renders in reconnecting state', () => {
    const mock = require('../../hooks/useConnectionStatus');
    mock.useConnectionStatus.mockReturnValueOnce({state: 'reconnecting', latencyMs: null, retryCount: 2});
    expect(() => render(<ConnectionStatusBar />)).not.toThrow();
  });
});
