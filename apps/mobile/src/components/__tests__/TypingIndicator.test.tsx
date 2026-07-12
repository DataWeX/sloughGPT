import React from 'react';
import {render} from '../../test-utils';
import {TypingIndicator} from '../TypingIndicator';

describe('TypingIndicator', () => {
  it('renders nothing when not visible', () => {
    expect(() => render(<TypingIndicator visible={false} />)).not.toThrow();
  });

  it('renders dots when visible', () => {
    expect(() => render(<TypingIndicator visible={true} />)).not.toThrow();
  });
});
