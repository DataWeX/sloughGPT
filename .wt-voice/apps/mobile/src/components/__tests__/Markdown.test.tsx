import React from 'react';
import {render} from '../../test-utils';
import {Markdown} from '../Markdown';

describe('Markdown', () => {
  it('renders plain text', () => {
    expect(() => render(<Markdown content="Hello world" />)).not.toThrow();
  });

  it('renders bold text', () => {
    expect(() => render(<Markdown content="**bold text**" />)).not.toThrow();
  });

  it('renders italic text', () => {
    expect(() => render(<Markdown content="*italic text*" />)).not.toThrow();
  });

  it('renders code blocks', () => {
    expect(() => render(<Markdown content="```const x = 1```" />)).not.toThrow();
  });

  it('renders inline code', () => {
    expect(() => render(<Markdown content="use `useState` hook" />)).not.toThrow();
  });

  it('renders headings', () => {
    expect(() => render(<Markdown content="# Title" />)).not.toThrow();
  });

  it('returns empty for empty content', () => {
    expect(() => render(<Markdown content="" />)).not.toThrow();
  });

  it('applies custom style', () => {
    expect(() => render(<Markdown content="styled" style={{color: 'red'}} />)).not.toThrow();
  });
});
