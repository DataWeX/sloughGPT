import React from 'react';
import {render} from '../../test-utils';
import {StatusBadge} from '../StatusBadge';

describe('StatusBadge', () => {
  it('renders with default variant', () => {
    expect(() => render(<StatusBadge label="Active" />)).not.toThrow();
  });

  it('renders with success variant', () => {
    expect(() => render(<StatusBadge label="Online" variant="success" />)).not.toThrow();
  });

  it('renders with warning variant', () => {
    expect(() => render(<StatusBadge label="Slow" variant="warning" />)).not.toThrow();
  });

  it('renders with error variant', () => {
    expect(() => render(<StatusBadge label="Failed" variant="error" />)).not.toThrow();
  });

  it('renders with info variant', () => {
    expect(() => render(<StatusBadge label="Info" variant="info" />)).not.toThrow();
  });
});
