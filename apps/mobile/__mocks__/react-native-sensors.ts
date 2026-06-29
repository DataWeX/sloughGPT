const mockObservable = {
  subscribe: jest.fn(({next}: any) => {
    const interval = setInterval(() => {
      next({x: 0.1, y: 0.2, z: 9.8});
    }, 100);
    return {unsubscribe: () => clearInterval(interval)};
  }),
  pipe: jest.fn(function(this: any) { return this; }),
};

export const accelerometer = jest.fn(() => mockObservable);
export const gyroscope = jest.fn(() => mockObservable);
