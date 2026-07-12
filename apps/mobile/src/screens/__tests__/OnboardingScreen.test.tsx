import React from 'react';
import {render} from '@/test-utils';
import AsyncStorage from '@react-native-async-storage/async-storage';
import {isFirstLaunch, markOnboarded} from '../OnboardingScreen';

jest.mock('../../services/haptics', () => ({
  triggerHaptic: jest.fn(),
}));

beforeEach(() => {
  jest.clearAllMocks();
  AsyncStorage.clear();
});

describe('OnboardingScreen', () => {
  it('renders without crashing', () => {
    const {OnboardingScreen} = require('../OnboardingScreen');
    expect(() => render(<OnboardingScreen onComplete={jest.fn()} />)).not.toThrow();
  });
});

describe('isFirstLaunch / markOnboarded', () => {
  it('isFirstLaunch returns true on fresh install', async () => {
    expect(await isFirstLaunch()).toBe(true);
  });

  it('isFirstLaunch returns false after marking', async () => {
    await markOnboarded();
    expect(await isFirstLaunch()).toBe(false);
  });

  it('markOnboarded persists to AsyncStorage', async () => {
    await markOnboarded();
    const val = await AsyncStorage.getItem('@sloughgpt/onboarded');
    expect(val).toBe('true');
  });
});
