import React from 'react';
import {render} from '@testing-library/react-native';
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
  it('renders first step title', () => {
    const {getByText} = render(<OnboardingScreen onComplete={jest.fn()} />);
    expect(getByText('Chat with AI')).toBeTruthy();
  });

  it('renders skip button', () => {
    const {getByText} = render(<OnboardingScreen onComplete={jest.fn()} />);
    expect(getByText('Skip')).toBeTruthy();
  });

  it('renders Next button', () => {
    const {getByText} = render(<OnboardingScreen onComplete={jest.fn()} />);
    expect(getByText('Next')).toBeTruthy();
  });

  it('skip calls onComplete and marks onboarded', async () => {
    const onComplete = jest.fn();
    const {getByText} = render(<OnboardingScreen onComplete={onComplete} />);
    getByText('Skip').props.onPress();
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
