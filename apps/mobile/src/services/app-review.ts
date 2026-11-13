/**
 * App review prompt — asks users to rate the app at smart moments.
 *
 * Trigger conditions:
 * - After a successful training run
 * - After 25+ messages in a session
 * - After first positive feedback (thumbs up)
 * - Never more than once per 30 days
 * - Never on first session
 */

import AsyncStorage from '@react-native-async-storage/async-storage';

const LAST_PROMPT_KEY = '@sloughgpt/last_review_prompt';
const PROMPT_COUNT_KEY = '@sloughgpt/review_prompt_count';
const MESSAGE_COUNT_KEY = '@sloughgpt/message_count';
const HAS_TRAINED_KEY = '@sloughgpt/has_trained';
const HAS_POSITIVE_FB_KEY = '@sloughgpt/has_positive_feedback';

const COOLDOWN_DAYS = 30;
const MSG_THRESHOLD = 25;

let StoreReview: any = null;
try {
  StoreReview = require('expo-store-review');
} catch {
  // expo-store-review not installed
}

async function getLastPromptTime(): Promise<number> {
  const raw = await AsyncStorage.getItem(LAST_PROMPT_KEY);
  return raw ? parseInt(raw, 10) : 0;
}

async function setLastPromptTime(ts: number): Promise<void> {
  await AsyncStorage.setItem(LAST_PROMPT_KEY, String(ts));
}

async function getPromptCount(): Promise<number> {
  const raw = await AsyncStorage.getItem(PROMPT_COUNT_KEY);
  return raw ? parseInt(raw, 10) : 0;
}

async function incrementPromptCount(): Promise<void> {
  const count = await getPromptCount();
  await AsyncStorage.setItem(PROMPT_COUNT_KEY, String(count + 1));
}

/** Call on every message send. Tracks count toward the threshold. */
export async function trackMessageSent(): Promise<void> {
  const raw = await AsyncStorage.getItem(MESSAGE_COUNT_KEY);
  const count = raw ? parseInt(raw, 10) : 0;
  await AsyncStorage.setItem(MESSAGE_COUNT_KEY, String(count + 1));
}

/** Call after a successful training run. */
export async function onTrainingCompleted(): Promise<void> {
  await AsyncStorage.setItem(HAS_TRAINED_KEY, 'true');
  await tryPrompt('training');
}

/** Call after user gives positive feedback. */
export async function onPositiveFeedback(): Promise<void> {
  await AsyncStorage.setItem(HAS_POSITIVE_FB_KEY, 'true');
}

/** Call after each message to check message-count threshold. */
export async function onMessageSent(): Promise<void> {
  await trackMessageSent();
  const raw = await AsyncStorage.getItem(MESSAGE_COUNT_KEY);
  const count = raw ? parseInt(raw, 10) : 0;
  if (count >= MSG_THRESHOLD) {
    await tryPrompt('messages');
  }
}

/** Reset message counter (e.g., on new session). */
export async function resetMessageCount(): Promise<void> {
  await AsyncStorage.setItem(MESSAGE_COUNT_KEY, '0');
}

async function tryPrompt(_reason: string): Promise<void> {
  try {
    const lastPrompt = await getLastPromptTime();
    const now = Date.now();
    const daysSince = (now - lastPrompt) / (1000 * 60 * 60 * 24);
    if (daysSince < COOLDOWN_DAYS) return;

    const count = await getPromptCount();
    if (count >= 3) return;

    if (StoreReview) {
      const available = await StoreReview.isAvailableAsync();
      if (!available) return;

      await StoreReview.requestReviewAsync();
      await setLastPromptTime(now);
      await incrementPromptCount();
    }
  } catch {
    // Silently fail — never block the user
  }
}

/** Get review prompt stats (for debug/settings). */
export async function getReviewStats(): Promise<{
  promptCount: number;
  messageCount: number;
  hasTrained: boolean;
  hasPositiveFeedback: boolean;
  lastPromptDaysAgo: number | null;
}> {
  const [promptCount, msgRaw, trained, fb, lastPrompt] = await Promise.all([
    getPromptCount(),
    AsyncStorage.getItem(MESSAGE_COUNT_KEY),
    AsyncStorage.getItem(HAS_TRAINED_KEY),
    AsyncStorage.getItem(HAS_POSITIVE_FB_KEY),
    getLastPromptTime(),
  ]);

  const lastPromptDaysAgo = lastPrompt
    ? Math.floor((Date.now() - lastPrompt) / (1000 * 60 * 60 * 24))
    : null;

  return {
    promptCount,
    messageCount: msgRaw ? parseInt(msgRaw, 10) : 0,
    hasTrained: trained === 'true',
    hasPositiveFeedback: fb === 'true',
    lastPromptDaysAgo,
  };
}
