import React from 'react';
import {render} from '../../test-utils';
import {Skeleton, SkeletonCard, SkeletonList} from '../Skeleton';

describe('Skeleton', () => {
  it('renders with default props', () => {
    expect(() => render(<Skeleton />)).not.toThrow();
  });

  it('renders with custom dimensions', () => {
    expect(() => render(<Skeleton width={200} height={40} borderRadius={8} />)).not.toThrow();
  });
});

describe('SkeletonCard', () => {
  it('renders with default lines', () => {
    expect(() => render(<SkeletonCard />)).not.toThrow();
  });

  it('renders with custom line count', () => {
    expect(() => render(<SkeletonCard lines={5} />)).not.toThrow();
  });
});

describe('SkeletonList', () => {
  it('renders multiple cards', () => {
    expect(() => render(<SkeletonList count={3} />)).not.toThrow();
  });
});
