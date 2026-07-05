import React from 'react';
import {render} from '@testing-library/react-native';
import {Skeleton, SkeletonCard, SkeletonList} from '../Skeleton';

describe('Skeleton', () => {
  it('renders with default props', () => {
    const {toJSON} = render(<Skeleton />);
    expect(toJSON()).toBeTruthy();
  });

  it('renders with custom dimensions', () => {
    const {toJSON} = render(<Skeleton width={200} height={40} borderRadius={8} />);
    expect(toJSON()).toBeTruthy();
  });
});

describe('SkeletonCard', () => {
  it('renders with default lines', () => {
    const {toJSON} = render(<SkeletonCard />);
    expect(toJSON()).toBeTruthy();
  });

  it('renders with custom line count', () => {
    const {toJSON} = render(<SkeletonCard lines={5} />);
    expect(toJSON()).toBeTruthy();
  });
});

describe('SkeletonList', () => {
  it('renders multiple cards', () => {
    const {toJSON} = render(<SkeletonList count={3} />);
    expect(toJSON()).toBeTruthy();
  });
});
