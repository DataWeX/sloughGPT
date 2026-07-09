import React from 'react';
import {Text} from 'react-native';
import {render, waitFor} from '@testing-library/react-native';
import * as apiClient from '../../services/api-client';

jest.mock('../../services/api-client');

const mockGetApiUrl = apiClient.getApiUrl as jest.Mock;

// eslint-disable-next-line @typescript-eslint/no-var-requires
const {useOnlineStatus} = require('../useOnlineStatus');

function HookDisplay() {
  const val = useOnlineStatus();
  return <Text testID="val">{String(val)}</Text>;
}

beforeEach(() => {
  jest.clearAllMocks();
  jest.restoreAllMocks();
  mockGetApiUrl.mockResolvedValue('http://localhost:8000');
});

afterEach(() => {
  jest.restoreAllMocks();
});

describe('useOnlineStatus', () => {
  it('returns true when health check succeeds', async () => {
    jest.spyOn(global, 'fetch').mockResolvedValue({ok: true} as Response);
    const {getByTestId} = await render(<HookDisplay />);
    await waitFor(() => {
      expect(getByTestId('val').children[0]).toBe('true');
    });
  });

  it('returns false when health check fails with non-ok', async () => {
    jest.spyOn(global, 'fetch').mockResolvedValue({ok: false} as Response);
    const {getByTestId} = await render(<HookDisplay />);
    await waitFor(() => {
      expect(getByTestId('val').children[0]).toBe('false');
    });
  });

  it('returns false when fetch throws', async () => {
    jest.spyOn(global, 'fetch').mockRejectedValue(new Error('fail'));
    const {getByTestId} = await render(<HookDisplay />);
    await waitFor(() => {
      expect(getByTestId('val').children[0]).toBe('false');
    });
  });

  it('returns false when getApiUrl throws', async () => {
    mockGetApiUrl.mockRejectedValue(new Error('fail'));
    const {getByTestId} = await render(<HookDisplay />);
    await waitFor(() => {
      expect(getByTestId('val').children[0]).toBe('false');
    });
  });

  it('sets up polling interval', async () => {
    const spy = jest.spyOn(global, 'setInterval');
    jest.spyOn(global, 'fetch').mockResolvedValue({ok: true} as Response);
    await render(<HookDisplay />);
    expect(spy).toHaveBeenCalledWith(expect.any(Function), 10000);
  });

  it('calls clearInterval on unmount', async () => {
    const spy = jest.spyOn(global, 'clearInterval');
    jest.spyOn(global, 'fetch').mockResolvedValue({ok: true} as Response);
    const {unmount} = await render(<HookDisplay />);
    await unmount();
    expect(spy).toHaveBeenCalled();
  });
});
