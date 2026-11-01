jest.mock('../haptics');

import {toast, Toast} from '../toast';

beforeEach(() => {
  jest.useFakeTimers();
  // Clear all toasts
  toast.clear();
});

afterEach(() => {
  jest.useRealTimers();
});

describe('toast', () => {
  it('success adds a toast', () => {
    const id = toast.success('Done');
    expect(toast.getToasts()).toHaveLength(1);
    expect(toast.getToasts()[0].message).toBe('Done');
    expect(toast.getToasts()[0].type).toBe('success');
  });

  it('error adds a toast', () => {
    toast.error('Failed');
    expect(toast.getToasts()[0].type).toBe('error');
  });

  it('info adds a toast', () => {
    toast.info('Heads up');
    expect(toast.getToasts()[0].type).toBe('info');
  });

  it('warn adds a toast', () => {
    toast.warn('Caution');
    expect(toast.getToasts()[0].type).toBe('warn');
  });

  it('dismiss removes a toast by id', () => {
    const id = toast.success('Hi');
    expect(toast.getToasts()).toHaveLength(1);
    toast.dismiss(id);
    expect(toast.getToasts()).toHaveLength(0);
  });

  it('clear removes all toasts', () => {
    toast.success('A');
    toast.error('B');
    expect(toast.getToasts()).toHaveLength(2);
    toast.clear();
    expect(toast.getToasts()).toHaveLength(0);
  });

  it('limits to 5 visible toasts', () => {
    for (let i = 0; i < 10; i++) toast.success(`#${i}`);
    expect(toast.getToasts()).toHaveLength(5);
  });

  it('auto-dismisses after duration', () => {
    toast.success('Bye', 1000);
    expect(toast.getToasts()).toHaveLength(1);
    jest.advanceTimersByTime(1000);
    expect(toast.getToasts()).toHaveLength(0);
  });

  it('subscribe notifies on changes', () => {
    const fn = jest.fn();
    const unsub = toast.subscribe(fn);
    toast.success('Hi');
    expect(fn).toHaveBeenCalled();
    expect(fn.mock.calls[0][0]).toHaveLength(1);
    unsub();
  });

  it('subscribe returns unsubscribe function', () => {
    const fn = jest.fn();
    const unsub = toast.subscribe(fn);
    unsub();
    toast.success('Hi');
    expect(fn).not.toHaveBeenCalled();
  });
});
