import React from 'react';
import {Text} from 'react-native';
import {render} from '@testing-library/react-native';
import * as apiClient from '../../services/api-client';

jest.mock('../../services/api-client');

const mockGetApiUrl = apiClient.getApiUrl as jest.Mock;

// eslint-disable-next-line @typescript-eslint/no-var-requires
const {useConnectionStatus} = require('../useConnectionStatus');

function Display() {
  const info = useConnectionStatus();
  return <Text testID="val">{JSON.stringify(info)}</Text>;
}

beforeEach(() => {
  jest.clearAllMocks();
  mockGetApiUrl.mockResolvedValue('http://localhost:8000');
});

describe('useConnectionStatus timers', () => {
  it('sets up polling interval', async () => {
    jest.useFakeTimers();
    const spy = jest.spyOn(global, 'setInterval');
    jest
      .spyOn(global, 'fetch')
      .mockResolvedValue({ok: true, json: async () => ({})} as Response);
    const {unmount} = await render(<Display />);
    expect(spy).toHaveBeenCalledWith(expect.any(Function), 8000);
    await unmount();
    jest.runOnlyPendingTimers();
    jest.useRealTimers();
  });

  it('clears interval on unmount', async () => {
    jest.useFakeTimers();
    const spy = jest.spyOn(global, 'clearInterval');
    jest
      .spyOn(global, 'fetch')
      .mockResolvedValue({ok: true, json: async () => ({})} as Response);
    const {unmount} = await render(<Display />);
    await unmount();
    expect(spy).toHaveBeenCalled();
    jest.runOnlyPendingTimers();
    jest.useRealTimers();
  });
});
