import React from 'react';
import {render} from '../../test-utils';
import {QuickPromptPicker} from '../QuickPromptPicker';

jest.mock('../../services/quick-prompts', () => ({
  getQuickPromptsByCategory: jest.fn(() => Promise.resolve([])),
  addQuickPrompt: jest.fn(() => Promise.resolve('new-id')),
  deleteQuickPrompt: jest.fn(() => Promise.resolve()),
}));

describe('QuickPromptPicker', () => {
  it('renders nothing when not visible', () => {
    expect(() =>
      render(<QuickPromptPicker visible={false} onClose={jest.fn()} onSelect={jest.fn()} />),
    ).not.toThrow();
  });

  it('renders when visible', () => {
    expect(() =>
      render(<QuickPromptPicker visible={true} onClose={jest.fn()} onSelect={jest.fn()} />),
    ).not.toThrow();
  });
});
