import React from 'react';
import {render} from '../../test-utils';
import {ScreenShell} from '../ScreenShell';

describe('ScreenShell', () => {
  it('renders with title and children', () => {
    expect(() =>
      render(
        <ScreenShell title="Settings">
          <></>
        </ScreenShell>,
      ),
    ).not.toThrow();
  });

  it('renders without scroll', () => {
    expect(() =>
      render(
        <ScreenShell title="Settings" scroll={false}>
          <></>
        </ScreenShell>,
      ),
    ).not.toThrow();
  });

  it('renders with pull-to-refresh', () => {
    expect(() =>
      render(
        <ScreenShell title="Chat" refreshing={false} onRefresh={jest.fn()}>
          <></>
        </ScreenShell>,
      ),
    ).not.toThrow();
  });
});
