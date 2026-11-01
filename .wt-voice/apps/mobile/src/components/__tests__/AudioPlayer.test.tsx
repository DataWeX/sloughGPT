import React from 'react';
import {render} from '../../test-utils';
import AsyncStorage from '@react-native-async-storage/async-storage';
import {AudioPlayer, formatDuration} from '../AudioPlayer';

jest.mock('expo-av', () => ({
  Audio: {
    Sound: {
      createAsync: jest.fn().mockResolvedValue({
        sound: {
          playAsync: jest.fn().mockResolvedValue(undefined),
          pauseAsync: jest.fn().mockResolvedValue(undefined),
          unloadAsync: jest.fn().mockResolvedValue(undefined),
          getStatusAsync: jest.fn().mockResolvedValue({
            isLoaded: true,
            durationMillis: 10000,
            positionMillis: 0,
            didJustFinish: false,
          }),
        },
      }),
    },
    setAudioModeAsync: jest.fn().mockResolvedValue(undefined),
  },
}), {virtual: true});

const API_BASE = 'http://localhost:8000';

beforeEach(async () => {
  jest.clearAllMocks();
  await AsyncStorage.clear();
  await AsyncStorage.setItem('@sloughgpt/api-url', API_BASE);
});

describe('formatDuration', () => {
  it('formats 0ms as 0:00', () => {
    expect(formatDuration(0)).toBe('0:00');
  });

  it('formats 5000ms as 0:05', () => {
    expect(formatDuration(5000)).toBe('0:05');
  });

  it('formats 90000ms as 1:30', () => {
    expect(formatDuration(90000)).toBe('1:30');
  });

  it('formats 605000ms as 10:05', () => {
    expect(formatDuration(605000)).toBe('10:05');
  });

  it('formats negative ms as 0:00', () => {
    expect(formatDuration(-100)).toBe('0:00');
  });
});

describe('AudioPlayer', () => {
  it('renders fallback "voice" badge when no URL props given', async () => {
    const r = await render(<AudioPlayer />);
    expect(r.getByText('voice')).toBeTruthy();
  });

  it('renders play button and duration when audioUrl is provided', async () => {
    const r = await render(<AudioPlayer audioUrl="http://example.com/audio.m4a" durationMs={5000} />);
    expect(await r.findByText('▶')).toBeTruthy();
    expect(await r.findByText('0:05')).toBeTruthy();
  });

  it('renders play button and duration when audioPath is provided', async () => {
    const r = await render(<AudioPlayer audioPath="session-1/msg-1.m4a" durationMs={3000} />);
    expect(await r.findByText('▶')).toBeTruthy();
    expect(await r.findByText('0:03')).toBeTruthy();
  });

  it('renders duration in m:ss format for longer audio', async () => {
    const r = await render(<AudioPlayer audioUrl="http://test.com/a.m4a" durationMs={905000} />);
    expect(await r.findByText('15:05')).toBeTruthy();
  });

  it('hides duration text when durationMs is 0 and not playing', async () => {
    const r = await render(<AudioPlayer audioUrl="http://test.com/a.m4a" />);
    expect(r.queryByText('0:00')).toBeNull();
  });
});
