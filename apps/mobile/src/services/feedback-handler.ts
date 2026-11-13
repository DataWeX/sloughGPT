/**
 * UI feedback handler — unified haptic + sound for common app events.
 *
 * Replaces scattered `triggerHaptic('x') + sounds.y()` calls with a single
 * handler that maps semantic UI events to the correct haptic + sound combo.
 *
 * Usage:
 *   import {feedback} from '../services/feedback-handler';
 *   feedback.tap();       // button press, toggle, nav
 *   feedback.send();      // message sent
 *   feedback.receive();   // message received
 *   feedback.success();   // action completed
 *   feedback.error();     // action failed
 *   feedback.delete();    // item deleted
 *   feedback.warning();   // caution needed
 *   feedback.select();    // picker / segmented control
 */

import {triggerHaptic, type HapticType} from './haptics';
import {sounds} from './sounds';

/** Semantic UI events mapped to haptic + sound. */
interface FeedbackEvent {
  haptic: HapticType;
  sound: 'send' | 'receive' | 'error' | 'delete' | null;
}

const EVENT_MAP: Record<string, FeedbackEvent> = {
  /** Button press, toggle, navigation tap. */
  tap: {haptic: 'light', sound: null},
  /** Message sent by user. */
  send: {haptic: 'medium', sound: 'send'},
  /** Assistant message received. */
  receive: {haptic: 'light', sound: 'receive'},
  /** Action completed successfully (save, copy, export). */
  success: {haptic: 'success', sound: 'receive'},
  /** Action failed. */
  error: {haptic: 'error', sound: 'error'},
  /** Item deleted / removed. */
  delete: {haptic: 'medium', sound: 'delete'},
  /** Caution / non-blocking warning. */
  warning: {haptic: 'medium', sound: null},
  /** Selection change (picker, segmented control). */
  select: {haptic: 'selection', sound: null},
  /** Heavy confirmation (destructive action preview). */
  confirm: {haptic: 'heavy', sound: null},
};

export type FeedbackEventName = keyof typeof EVENT_MAP;

class FeedbackHandler {
  private _enabled = true;

  /** Enable or disable all feedback (haptic + sound). */
  setEnabled(enabled: boolean) {
    this._enabled = enabled;
    sounds.setEnabled(enabled);
  }

  /** Check if feedback is enabled. */
  isEnabled(): boolean {
    return this._enabled;
  }

  /** Fire a named feedback event. No-op if disabled. */
  async fire(event: FeedbackEventName): Promise<void> {
    if (!this._enabled) return;
    const config = EVENT_MAP[event];
    if (!config) return;

    // Fire haptic (fire-and-forget — don't await)
    triggerHaptic(config.haptic).catch(() => {});

    // Fire sound (fire-and-forget)
    if (config.sound) {
      sounds[config.sound]().catch(() => {});
    }
  }

  // ── Convenience methods ────────────────────────────────────────────────

  tap() { return this.fire('tap'); }
  send() { return this.fire('send'); }
  receive() { return this.fire('receive'); }
  success() { return this.fire('success'); }
  error() { return this.fire('error'); }
  delete() { return this.fire('delete'); }
  warning() { return this.fire('warning'); }
  select() { return this.fire('select'); }
  confirm() { return this.fire('confirm'); }
}

/**
 * Singleton UI feedback handler.
 *
 * Fires haptic + sound in parallel (fire-and-forget) so calling code
 * never blocks on audio/hardware.
 */
export const feedback = new FeedbackHandler();
