import * as React from 'react';

/**
 * Manual jest mock for zustand (auto-applied to all suites — node-module rule).
 *
 * Faithful to zustand semantics the tests rely on:
 * - `state` is the single live source (initial factory result + every setState)
 * - selectors read that live state — never the stale initial `api` snapshot
 * - hooks subscribe via useSyncExternalStore, so components re-render on change
 *   (journey tests assert store-driven UI updates)
 * - selector results are stabilized with shallow equality (zustand v4-style):
 *   selectors like `s => s.operations.filter(...)` return fresh-but-equal
 *   arrays each call, which would otherwise loop React's update-depth guard.
 */
export const create = jest.fn((fn: any) => {
  let state: any = {};
  const listeners = new Set<(s: any) => void>();

  const setState = (partial: any) => {
    const next = typeof partial === 'function' ? partial(state) : partial;
    state = {...state, ...next};
    listeners.forEach(l => l(state));
  };

  const getState = () => state;
  const subscribe = (listener: (s: any) => void) => {
    listeners.add(listener);
    return () => listeners.delete(listener);
  };

  // Call factory to get initial state + actions, merge into live state
  const api = fn(setState, getState);
  state = {...state, ...api};

  const useStore = (selector?: (s: any) => any) => {
    const cache = React.useRef<{stateRef: any; value: any}>({stateRef: null, value: undefined});
    const getSnapshot = () => {
      const next = selector ? selector(state) : state;
      const c = cache.current;
      if (c.stateRef === state && shallowEqual(c.value, next)) {
        return c.value;
      }
      c.stateRef = state;
      c.value = next;
      return next;
    };
    return React.useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
  };

  Object.assign(useStore, api, {getState, setState, subscribe, getInitialState: getState});
  return useStore;
});

function shallowEqual(a: any, b: any): boolean {
  if (Object.is(a, b)) return true;
  if (typeof a !== 'object' || typeof b !== 'object' || a === null || b === null) return false;
  if (Array.isArray(a) !== Array.isArray(b)) return false;
  const keysA = Object.keys(a);
  const keysB = Object.keys(b);
  if (keysA.length !== keysB.length) return false;
  return keysA.every(k => Object.is(a[k], b[k]));
}
