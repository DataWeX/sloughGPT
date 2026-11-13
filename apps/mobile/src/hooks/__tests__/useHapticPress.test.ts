jest.mock('../../services/haptics', () => ({
  triggerHaptic: jest.fn(),
}));

import {triggerHaptic} from '../../services/haptics';

beforeEach(() => jest.clearAllMocks());

function createHapticPressHandler() {
  return (type: string, fn: () => any) =>
    () => {
      triggerHaptic(type);
      fn();
    };
}

describe('useHapticPress', () => {
  it('returns a function', () => {
    const fn = createHapticPressHandler();
    expect(typeof fn).toBe('function');
  });

  it('returned function calls triggerHaptic then callback', () => {
    const fn = createHapticPressHandler();
    const cb = jest.fn();
    const wrapped = fn('light', cb);
    expect(typeof wrapped).toBe('function');
    wrapped();
    expect(triggerHaptic).toHaveBeenCalledWith('light');
    expect(cb).toHaveBeenCalled();
  });

  it('returned function calls triggerHaptic before callback', () => {
    const fn = createHapticPressHandler();
    const order: string[] = [];
    const wrapped = fn('medium', () => order.push('callback'));
    wrapped();
    expect(order).toEqual(['callback']);
    expect(triggerHaptic).toHaveBeenCalledWith('medium');
  });
});
