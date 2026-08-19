/**
 * Voice input service.
 * Records audio via expo-av and transcribes via backend.
 */

import {Alert, Linking} from 'react-native';
import {getApiUrl} from './api-client';

let Audio: any;
try {
  Audio = require('expo-av');
} catch {
  Audio = null;
}

const MIN_RECORDING_DURATION_MS = 2000;
const TRANSCRIBE_TIMEOUT_MS = 30000;

export interface VoiceRecording {
  uri: string;
  duration: number;
}

/**
 * Start recording audio. Returns a stop function that yields the recording.
 */
export async function startRecording(): Promise<{stop: () => Promise<VoiceRecording | null>}> {
  if (!Audio) {
    Alert.alert('Audio unavailable', 'expo-av is not installed on this device.');
    throw new Error('Audio recording not available.');
  }

  const permission = await Audio.Recording.requestPermissionsAsync();
  if (!permission.granted) {
    Alert.alert(
      'Microphone permission required',
      'Please enable microphone access in your device settings to record voice.',
      [
        {text: 'Cancel', style: 'cancel'},
        {
          text: 'Open Settings',
          onPress: () => Linking.openSettings().catch(() => {}),
        },
      ],
    );
    throw new Error('Microphone permission required');
  }

  const recording = new Audio.Recording();
  await recording.prepareToRecordAsync(Audio.RecordingOptionsPresets.HIGH_QUALITY);
  await recording.startAsync();

  const startTime = Date.now();

  return {
    stop: async (): Promise<VoiceRecording | null> => {
      try {
        await recording.stopAndUnloadAsync();
        const uri = recording.getURI();
        const duration = Math.round((Date.now() - startTime) / 1000);
        if (!uri) return null;
        if (duration * 1000 < MIN_RECORDING_DURATION_MS) {
          Alert.alert('Recording too short', 'Please record for at least 2 seconds.');
          return null;
        }
        return {uri, duration};
      } catch (e) {
        if (__DEV__) console.warn('[voice-input] stop recording failed:', e);
        return null;
      }
    },
  };
}

/**
 * Transcribe a voice recording via the backend.
 * Throws on network/server errors; returns '' when no speech detected.
 */
export async function transcribeAudio(uri: string): Promise<string> {
  const baseUrl = await getApiUrl();
  const formData = new FormData();
  formData.append('audio', {
    uri,
    type: 'audio/m4a',
    name: 'recording.m4a',
  } as any);

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), TRANSCRIBE_TIMEOUT_MS);

  try {
    const res = await fetch(`${baseUrl}/multimodal/transcribe`, {
      method: 'POST',
      body: formData,
      headers: {'Content-Type': 'multipart/form-data'},
      signal: controller.signal,
    });

    if (!res.ok) {
      throw new Error(`Transcription failed (${res.status})`);
    }

    const data = await res.json();
    return data.text || '';
  } catch (e: any) {
    if (e.name === 'AbortError') {
      throw new Error('Transcription timed out');
    }
    throw e;
  } finally {
    clearTimeout(timeout);
  }
}
