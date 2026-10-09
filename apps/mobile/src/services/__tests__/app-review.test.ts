jest.mock('expo-store-review', () => ({
  isAvailableAsync: jest.fn().mockResolvedValue(true),
  requestReviewAsync: jest.fn().mockResolvedValue(undefined),
}));

jest.mock('@react-native-async-storage/async-storage', () =>
  require('@react-native-async-storage/async-storage/jest/async-storage-mock'),
);

import AsyncStorage from '@react-native-async-storage/async-storage';

import {
  trackMessageSent,
  onTrainingCompleted,
  onPositiveFeedback,
  onMessageSent,
  resetMessageCount,
  getReviewStats,
} from '../app-review';

beforeEach(async () => {
  await AsyncStorage.clear();
  jest.clearAllMocks();
});

describe('app-review', () => {
  it('trackMessageSent increments count', async () => {
    await trackMessageSent();
    await trackMessageSent();
    const stats = await getReviewStats();
    expect(stats.messageCount).toBe(2);
  });

  it('resetMessageCount resets to 0', async () => {
    await trackMessageSent();
    await trackMessageSent();
    await resetMessageCount();
    const stats = await getReviewStats();
    expect(stats.messageCount).toBe(0);
  });

  it('onPositiveFeedback sets flag', async () => {
    await onPositiveFeedback();
    const stats = await getReviewStats();
    expect(stats.hasPositiveFeedback).toBe(true);
  });

  it('onTrainingCompleted sets flag', async () => {
    await onTrainingCompleted();
    const stats = await getReviewStats();
    expect(stats.hasTrained).toBe(true);
  });

  it('getReviewStats returns defaults', async () => {
    const stats = await getReviewStats();
    expect(stats.promptCount).toBe(0);
    expect(stats.messageCount).toBe(0);
    expect(stats.hasTrained).toBe(false);
    expect(stats.hasPositiveFeedback).toBe(false);
    expect(stats.lastPromptDaysAgo).toBeNull();
  });

  it('onMessageSent increments and triggers at threshold', async () => {
    for (let i = 0; i < 24; i++) {
      await onMessageSent();
    }
    let stats = await getReviewStats();
    expect(stats.messageCount).toBe(24);

    await onMessageSent();
    stats = await getReviewStats();
    expect(stats.messageCount).toBe(25);
  });
});
