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

  // Call factory to get initial state + actions
  const api = fn(setState, getState);
  // Merge api (initial state + actions) into state
  state = {...state, ...api};

  const useStore = (selector?: (s: any) => any) => {
    const s = {...state, ...api};
    return selector ? selector(s) : s;
  };

  Object.assign(useStore, api, {getState, setState, subscribe});
  return useStore;
});
