import { useState, useEffect, useRef } from 'react';
import { Audio } from 'expo-av';
import { Platform } from 'react-native';
import { Analytics, PerformanceTracker } from '../lib/analytics';

export interface UseVoiceInputReturn {
  isRecording: boolean;
  isProcessing: boolean;
  startRecording: () => Promise<void>;
  stopRecording: () => Promise<string | null>;
  cancelRecording: () => void;
  error: string | null;
}

export function useVoiceInput(): UseVoiceInputReturn {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const recordingRef = useRef<Audio.Recording | null>(null);

  useEffect(() => {
    return () => {
      if (recordingRef.current) {
        recordingRef.current.stopAndUnloadAsync().catch(() => {});
      }
    };
  }, []);

  const startRecording = async () => {
    try {
      setError(null);
      
      const permission = await Audio.requestPermissionsAsync();
      if (!permission.granted) {
        setError('Microphone permission denied');
        Analytics.trackEvent('voice_permission_denied');
        return;
      }

      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
      });

      const { recording } = await Audio.Recording.createAsync(
        Audio.RecordingOptionsPresets.HIGH_QUALITY
      );

      recordingRef.current = recording;
      setIsRecording(true);
      Analytics.trackEvent('voice_recording_started');
    } catch (err) {
      setError('Failed to start recording');
      PerformanceTracker.trackError(err as Error, { context: 'voice_start_recording' });
    }
  };

  const stopRecording = async (): Promise<string | null> => {
    if (!recordingRef.current) return null;

    try {
      setIsRecording(false);
      setIsProcessing(true);

      await recordingRef.current.stopAndUnloadAsync();
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: false,
      });

      const uri = recordingRef.current.getURI();
      recordingRef.current = null;

      if (!uri) {
        setError('No recording URI');
        return null;
      }

      // For now, we'll just return the URI
      // In production, you'd send this to a speech-to-text API
      Analytics.trackEvent('voice_recording_completed', { uri });
      
      // TODO: Integrate with speech-to-text API
      // For demo purposes, we'll just return a placeholder
      const transcribedText = '[Voice input - integrate with speech-to-text API]';
      
      setIsProcessing(false);
      return transcribedText;
    } catch (err) {
      setError('Failed to stop recording');
      setIsProcessing(false);
      PerformanceTracker.trackError(err as Error, { context: 'voice_stop_recording' });
      return null;
    }
  };

  const cancelRecording = () => {
    if (recordingRef.current) {
      recordingRef.current.stopAndUnloadAsync().catch(() => {});
      recordingRef.current = null;
    }
    setIsRecording(false);
    setIsProcessing(false);
    Analytics.trackEvent('voice_recording_cancelled');
  };

  return {
    isRecording,
    isProcessing,
    startRecording,
    stopRecording,
    cancelRecording,
    error,
  };
}
