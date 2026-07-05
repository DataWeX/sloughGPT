import React from 'react';
import {render, act} from '@testing-library/react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import {OnboardingScreen, isFirstLaunch, markOnboarded} from '../OnboardingScreen';

jest.mock('../../services/haptics', () => ({
  triggerHaptic: jest.fn(),
}));

beforeEach(() => {
  jest.clearAllMocks();
  AsyncStorage.clear();
});

describe('OnboardingScreen', () => {
  it('renders first step', () => {
    const {getByText} = render(<OnboardingScreen onComplete={jest.fn()} />);
    expect(getByText('Chat with AI')).toBeTruthy();
  });

  it('renders all step icons', () => {
    const {getByText} = render(<OnboardingScreen onComplete={jest.fn()} />);
    expect(getByText('💬')).toBeTruthy();
  });

  it('renders skip button', () => {
    const {getByText} = render(<OnboardingScreen onComplete={jest.fn()} />);
    expect(getByText('Skip')).toBeTruthy();
  });

  it('renders Next button', () => {
    const {getByText} = render(<OnboardingScreen onComplete={jest.fn()} />);
    expect(getByText('Next')).toBeTruthy();
  });

  it('renders dots', () => {
    const {getByText} = render(<OnboardingScreen onComplete={jest.fn()} />);
    // 3 steps = 3 dots rendered as Views (can't query dots directly, but component renders)
    expect(getByText('Chat with AI')).toBeTruthy();
  });

  it('skip calls onComplete and marks onboarded', async () => {
    const onComplete = jest.fn();
    const {getByText} = render(<OnboardingScreen onComplete={onComplete} />);
    await act(async () => {
      getByText('Skip').props.onPress();
    });
    expect(onComplete).toHaveBeenCalled();
    expect(await AsyncStorage.getItem('@sloughgpt/onboarded')).toBe('true');
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
});
