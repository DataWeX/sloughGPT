import React from 'react';
import {render} from '../../test-utils';
import {ToastContainer} from '../ToastContainer';

jest.mock('../../services/toast', () => ({
  toast: {
    subscribe: jest.fn(() => jest.fn()),
    getToasts: jest.fn(() => []),
  },
}));

jest.mock('../../services/haptics', () => ({triggerHaptic: jest.fn()}));

describe('ToastContainer', () => {
  it('renders nothing when no toasts', () => {
    expect(() => render(<ToastContainer />)).not.toThrow();
  });
});
