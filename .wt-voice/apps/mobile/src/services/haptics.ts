/**
 * Haptic feedback service.
 * Uses React Native's built-in InteractionManager for lightweight feedback.
 * Falls back gracefully on web or unsupported devices.
 */

import {Platform, InteractionManager} from 'react-native';

export type HapticType = 'light' | 'medium' | 'heavy' | 'success' | 'error' | 'selection';

let Haptics: any = null;
try {
  Haptics = require('expo-haptics');
} catch {
  // expo-haptics not installed — use fallback
}

export async function triggerHaptic(type: HapticType = 'light'): Promise<void> {
  if (Haptics) {
    try {
      switch (type) {
        case 'light':
          await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
          break;
        case 'medium':
          await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
          break;
        case 'heavy':
          await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Heavy);
          break;
        case 'success':
          await Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
          break;
        case 'error':
          await Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
          break;
        case 'selection':
          await Haptics.selectionAsync();
          break;
      }
      return;
    } catch {
      // fall through to native fallback
    }
  }

  // Native Android fallback via RN Feedback API
  if (Platform.OS === 'android') {
    try {
      const {Vibration} = require('react-native');
      switch (type) {
        case 'light':
          Vibration.vibrate(10);
          break;
        case 'medium':
          Vibration.vibrate(20);
          break;
        case 'heavy':
          Vibration.vibrate(40);
          break;
        case 'error':
          Vibration.vibrate([0, 10, 50, 10]);
          break;
        default:
          break;
      }
    } catch {
      // no vibration support
    }
  }

  // iOS without expo-haptics: no-op (UIFeedbackGenerator requires native module)
}
