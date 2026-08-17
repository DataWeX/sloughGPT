import {useCallback} from 'react';
import {triggerHaptic, type HapticType} from '../services/haptics';

export function useHapticPress() {
  return useCallback(
    (type: HapticType, fn: () => any) =>
      () => {
        triggerHaptic(type);
        fn();
      },
    [],
  );
}
