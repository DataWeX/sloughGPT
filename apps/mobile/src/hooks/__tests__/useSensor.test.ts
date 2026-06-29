import AsyncStorage from '@react-native-async-storage/async-storage';

// Mock react-native-sensors to throw (falls back to simulated)
jest.mock('react-native-sensors', () => {
  throw new Error('not installed');
});

// Mock activity-store
const mockPushReading = jest.fn();
jest.mock('../../stores/activity-store', () => ({
  useActivityStore: Object.assign(
    (selector?: (s: any) => any) => {
      const state = {pushReading: mockPushReading};
      return selector ? selector(state) : state;
    },
    {getState: () => ({pushReading: mockPushReading})},
  ),
}));

import {renderHook} from '@testing-library/react-native';
import {useSensor} from '../useSensor';

beforeEach(() => {
  jest.clearAllMocks();
});

describe('useSensor', () => {
  it('renders without crashing when active=true', () => {
    expect(() => {
      renderHook(() => useSensor(true));
    }).not.toThrow();
  });

  it('renders without crashing when active=false', () => {
    expect(() => {
      renderHook(() => useSensor(false));
    }).not.toThrow();
  });

  it('does not call pushReading when inactive', () => {
    renderHook(() => useSensor(false));
    expect(mockPushReading).not.toHaveBeenCalled();
  });
});
