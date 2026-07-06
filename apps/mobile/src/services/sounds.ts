/**
 * Message sound effects — lightweight audio feedback.
 * Maps events to haptic patterns + optional Audio API sounds.
 * Gracefully degrades to haptics-only when Audio API unavailable.
 */

import {Platform} from 'react-native';
import {triggerHaptic} from './haptics';

let Audio: any = null;
try {
  Audio = require('expo-av').Audio;
} catch {}

type SoundEvent = 'send' | 'receive' | 'error' | 'delete';

const SOUND_MAP: Record<SoundEvent, {haptic: string; audio?: string}> = {
  send: {haptic: 'medium'},
  receive: {haptic: 'light'},
  error: {haptic: 'error'},
  delete: {haptic: 'medium'},
};

let _enabled = true;

export function setSoundsEnabled(enabled: boolean) {
  _enabled = enabled;
}

export function areSoundsEnabled(): boolean {
  return _enabled;
}

export async function playSound(event: SoundEvent): Promise<void> {
  if (!_enabled) return;

  const config = SOUND_MAP[event];

  // Always do haptic
  triggerHaptic(config.haptic as any);

  // Try audio if available (system sounds)
  if (Audio && Platform.OS === 'ios') {
    try {
      const {sound} = await Audio.Sound.createAsync(
        {uri: _systemSoundUri(event)},
        {shouldPlay: true, volume: 0.3},
      );
      // Unload after playback
      setTimeout(() => sound.unloadAsync(), 1000);
    } catch {
      // Audio not available — haptic only is fine
    }
  }
}

function _systemSoundUri(event: SoundEvent): string {
  // iOS system sounds via private API — not reliable
  // Use a silent fallback so audio path doesn't crash
  return 'about:blank';
}

// Convenience exports
export const sounds = {
  send: () => playSound('send'),
  receive: () => playSound('receive'),
  error: () => playSound('error'),
  delete: () => playSound('delete'),
  setEnabled: setSoundsEnabled,
  isEnabled: areSoundsEnabled,
};
