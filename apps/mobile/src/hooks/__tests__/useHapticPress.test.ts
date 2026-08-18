jest.mock('../../services/haptics', () => ({
  triggerHaptic: jest.fn(),
}));

import {useHapticPress} from '../../hooks/useHapticPress';
import {triggerHaptic} from '../../services/haptics';

beforeEach(() => jest.clearAllMocks());

describe('useHapticPress', () => {
  it('returns a function', () => {
    const fn = useHapticPress();
    expect(typeof fn).toBe('function');
  });

  it('returned function calls triggerHaptic then callback', () => {
    const fn = useHapticPress();
    const cb = jest.fn();
    const wrapped = fn('light', cb);
    expect(typeof wrapped).toBe('function');
    wrapped();
    expect(triggerHaptic).toHaveBeenCalledWith('light');
    expect(cb).toHaveBeenCalled();
  });

  it('returned function returns callback result', () => {
    const fn = useHapticPress();
    const wrapped = fn('medium', () => 42);
    expect(wrapped()).toBe(42);
  });
});
