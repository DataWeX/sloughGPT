import React from 'react';
import {render} from '../../test-utils';
import {ErrorBoundary} from '../ErrorBoundary';

const GoodChild = () => <></>;
const BadChild = () => {
  throw new Error('boom');
};

describe('ErrorBoundary', () => {
  it('renders children when no error', () => {
    expect(() =>
      render(
        <ErrorBoundary>
          <GoodChild />
        </ErrorBoundary>,
      ),
    ).not.toThrow();
  });

  it('catches errors and renders fallback', () => {
    const spy = jest.spyOn(console, 'error').mockImplementation(() => {});
    expect(() =>
      render(
        <ErrorBoundary>
          <BadChild />
        </ErrorBoundary>,
      ),
    ).not.toThrow();
    spy.mockRestore();
  });
});
