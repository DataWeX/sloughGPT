import {toast, type Toast, type ToastType} from '../toast';

beforeEach(() => {
  toast.clear();
  jest.useFakeTimers();
});

afterEach(() => {
  jest.useRealTimers();
});

describe('toast service', () => {
  describe('add/remove', () => {
    it('success adds a toast', () => {
      const id = toast.success('Sent');
      expect(id).toMatch(/^toast-/);
      expect(toast.getToasts()).toHaveLength(1);
      expect(toast.getToasts()[0].type).toBe('success');
      expect(toast.getToasts()[0].message).toBe('Sent');
    });

    it('error adds a toast', () => {
      toast.error('Failed');
      expect(toast.getToasts()[0].type).toBe('error');
    });

    it('info adds a toast', () => {
      toast.info('Syncing');
      expect(toast.getToasts()[0].type).toBe('info');
    });

    it('warn adds a toast', () => {
      toast.warn('Slow connection');
      expect(toast.getToasts()[0].type).toBe('warn');
    });

    it('dismiss removes a toast', () => {
      const id = toast.success('Test');
      expect(toast.getToasts()).toHaveLength(1);
      toast.dismiss(id);
      expect(toast.getToasts()).toHaveLength(0);
    });

    it('clear removes all toasts', () => {
      toast.success('a');
      toast.error('b');
      toast.info('c');
      expect(toast.getToasts()).toHaveLength(3);
      toast.clear();
      expect(toast.getToasts()).toHaveLength(0);
    });

    it('max 5 visible toasts', () => {
      for (let i = 0; i < 8; i++) {
        toast.info(`msg ${i}`);
      }
      expect(toast.getToasts()).toHaveLength(5);
    });

    it('newest first', () => {
      toast.success('first');
      toast.error('second');
      const toasts = toast.getToasts();
      expect(toasts[0].message).toBe('second');
      expect(toasts[1].message).toBe('first');
    });
  });

  describe('auto-dismiss', () => {
    it('success auto-dismisses after 2500ms', () => {
      toast.success('auto');
      expect(toast.getToasts()).toHaveLength(1);
      jest.advanceTimersByTime(2500);
      expect(toast.getToasts()).toHaveLength(0);
    });

    it('error auto-dismisses after 4000ms', () => {
      toast.error('auto');
      jest.advanceTimersByTime(3999);
      expect(toast.getToasts()).toHaveLength(1);
      jest.advanceTimersByTime(1);
      expect(toast.getToasts()).toHaveLength(0);
    });

    it('info auto-dismisses after 2000ms', () => {
      toast.info('auto');
      jest.advanceTimersByTime(2000);
      expect(toast.getToasts()).toHaveLength(0);
    });

    it('warn auto-dismisses after 3500ms', () => {
      toast.warn('auto');
      jest.advanceTimersByTime(3500);
      expect(toast.getToasts()).toHaveLength(0);
    });

    it('custom duration overrides default', () => {
      toast.success('custom', 500);
      jest.advanceTimersByTime(499);
      expect(toast.getToasts()).toHaveLength(1);
      jest.advanceTimersByTime(1);
      expect(toast.getToasts()).toHaveLength(0);
    });
  });

  describe('subscribe', () => {
    it('receives updates', () => {
      const listener = jest.fn();
      const unsub = toast.subscribe(listener);
      toast.success('test');
      expect(listener).toHaveBeenCalled();
      expect(listener.mock.calls[0][0]).toHaveLength(1);
      unsub();
    });

    it('unsubscribe stops updates', () => {
      const listener = jest.fn();
      const unsub = toast.subscribe(listener);
      unsub();
      toast.success('test');
      expect(listener).toHaveBeenCalledTimes(0);
    });

    it('dismiss notifies listener', () => {
      const listener = jest.fn();
      toast.subscribe(listener);
      const id = toast.success('test');
      listener.mockClear();
      toast.dismiss(id);
      expect(listener).toHaveBeenCalledWith([]);
    });
  });

  describe('toast shape', () => {
    it('has required fields', () => {
      const id = toast.success('hello');
      const t = toast.getToasts()[0];
      expect(t.id).toBe(id);
      expect(t.type).toBe('success');
      expect(t.message).toBe('hello');
      expect(t.duration).toBe(2500);
      expect(t.timestamp).toBeGreaterThan(0);
    });
  });
});
