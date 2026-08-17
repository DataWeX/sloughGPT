import {useState, useRef, useCallback} from 'react';
import {triggerHaptic} from '../../services/haptics';
import {api} from '../../services/api-client';
import {startRecording, transcribeAudio} from '../../services/voice-input';
import {toast} from '../../services/toast';
import {useChatStore} from '../../stores/chat-store';
import type {Message} from '../../types';

export function useChatVoice(
  sendMessage: (text: string) => void,
  activeSessionId: string | null,
  createSession: () => Promise<any>,
) {
  const [isRecording, setIsRecording] = useState(false);
  const [voiceMessageMode, setVoiceMessageMode] = useState(false);
  const recordingStopRef = useRef<(() => Promise<{uri: string; duration: number} | null>) | null>(null);
  const voiceTimerRef = useRef<{start: number} | null>(null);

  const handleVoice = useCallback(async () => {
    if (isRecording) {
      const stop = recordingStopRef.current;
      if (stop) {
        const recording = await stop();
        setIsRecording(false);
        recordingStopRef.current = null;
        if (recording) {
          if (voiceMessageMode) {
            let sid = activeSessionId;
            if (!sid) {
              await createSession();
              sid = useChatStore.getState().activeSessionId;
            }
            if (sid) {
              try {
                const result = await api.sendVoiceMessage(sid, recording.uri, recording.duration);
                if (result.message_id) {
                  const voiceMsg: Message = {
                    id: Date.now().toString(36) + Math.random().toString(36).slice(2, 5),
                    role: 'user',
                    content: '🎤 Voice message',
                    timestamp: Date.now(),
                    audio_path: result.audio_path,
                    audio_duration_ms: recording.duration,
                    _voice: true,
                    status: 'sent',
                  };
                  useChatStore.setState(s => ({
                    messages: [...s.messages, voiceMsg],
                  }));
                  toast.success('Voice message sent');
                }
              } catch {
                toast.error('Failed to send voice message');
              }
            }
          } else {
            try {
              const text = await transcribeAudio(recording.uri);
              if (text) {
                sendMessage(text);
              } else {
                toast.warn('Could not transcribe audio');
              }
            } catch {
              toast.error('Transcription failed');
            }
          }
        }
      }
    } else {
      try {
        const {stop} = await startRecording();
        recordingStopRef.current = stop;
        setIsRecording(true);
        triggerHaptic('medium');
      } catch (e: any) {
        toast.error(e.message || 'Failed to start recording');
      }
    }
  }, [isRecording, sendMessage, voiceMessageMode, activeSessionId, createSession]);

  return {
    isRecording,
    setIsRecording,
    voiceMessageMode,
    setVoiceMessageMode,
    recordingStopRef,
    voiceTimerRef,
    handleVoice,
  };
}
