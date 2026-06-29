export const combineLatest = jest.fn((observables: any[]) => ({
  subscribe: jest.fn(({next}: any) => {
    const interval = setInterval(() => {
      next(observables.map(() => ({x: 0.1, y: 0.2, z: 9.8})));
    }, 100);
    return {unsubscribe: () => clearInterval(interval)};
  }),
  pipe: jest.fn(function(this: any) { return this; }),
}));

export const filter = jest.fn(() => (source: any) => source);
