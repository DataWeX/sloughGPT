import React from 'react';
import {render} from '../../test-utils';
import {LoadingScreen} from '../LoadingScreen';

describe('LoadingScreen', () => {
  it('renders without crashing', () => {
    expect(() => render(<LoadingScreen />)).not.toThrow();
  });

  it('renders with custom message', () => {
    expect(() => render(<LoadingScreen message="Loading models..." />)).not.toThrow();
  });
});
