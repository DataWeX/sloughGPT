/**
 * Voice input service.
 * Records audio via expo-av and transcribes via backend.
 */

import {Alert, Linking} from 'react-native';

let Audio: any;
try {
  Audio = require('expo-av');
} catch {
  Audio = null;
}

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
        return uri ? {uri, duration} : null;
      } catch {
        return null;
      }
    },
  };
}

/**
 * Transcribe a voice recording via the backend.
 */
export async function transcribeAudio(uri: string): Promise<string> {
  try {
    const {getApiUrl} = require('./api-client');
    const baseUrl = await getApiUrl();
    const formData = new FormData();
    formData.append('audio', {
      uri,
      type: 'audio/m4a',
      name: 'recording.m4a',
    } as any);

    const res = await fetch(`${baseUrl}/multimodal/transcribe`, {
      method: 'POST',
      body: formData,
      headers: {'Content-Type': 'multipart/form-data'},
    });

    if (res.ok) {
      const data = await res.json();
      return data.text || '';
    }
  } catch {
    // backend unavailable
  }

  return '';
}
