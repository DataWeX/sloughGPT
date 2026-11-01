import React from 'react';
import {render} from '../../test-utils';
import {ReasoningPanel} from '../ReasoningPanel';

describe('ReasoningPanel', () => {
  it('renders nothing when not visible', () => {
    expect(() => render(<ReasoningPanel visible={false} />)).not.toThrow();
  });

  it('renders reasoning indicator when visible', () => {
    expect(() => render(<ReasoningPanel visible={true} />)).not.toThrow();
  });
});
