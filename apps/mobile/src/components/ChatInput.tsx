import React, {useState, useRef, useEffect} from 'react';
import {
  View,
  TextInput,
  TouchableOpacity,
  Text,
  StyleSheet,
  Keyboard,
} from 'react-native';
import {colors, spacing, radii, typography} from '../theme';
import {triggerHaptic} from '../services/haptics';
import {QuickPromptPicker} from './QuickPromptPicker';
import {pickDocument, isTextFile, formatFileSize} from '../services/file-upload';
import {getDraft, saveDraft, clearDraft} from '../services/drafts';
import {toast} from '../services/toast';

interface Props {
  onSend: (text: string) => void;
  onImage?: () => void;
  onVoice?: () => void;
  onFile?: (content: string, name: string) => void;
  disabled?: boolean;
  onStop?: () => void;
  isRecording?: boolean;
  sessionId?: string | null;
  editText?: string | null;
  onCancelEdit?: () => void;
  /** When true, voice button sends audio as voice message instead of transcribing. */
  voiceMessageMode?: boolean;
  onVoiceMessageToggle?: () => void;
}

export function ChatInput({onSend, onImage, onVoice, onFile, disabled, onStop, isRecording, sessionId, editText, onCancelEdit, voiceMessageMode, onVoiceMessageToggle}: Props) {
  const [text, setText] = useState('');
  const [showPrompts, setShowPrompts] = useState(false);
  const inputRef = useRef<TextInput>(null);
  const draftTimerRef = useRef<NodeJS.Timeout | null>(null);

  // Load edit text when provided
  useEffect(() => {
    if (editText) {
      setText(editText);
      inputRef.current?.focus();
    }
  }, [editText]);

  // Load draft when session changes
  useEffect(() => {
    if (sessionId) {
      getDraft(sessionId).then(setText);
    }
  }, [sessionId]);

  // Auto-save draft as user types (debounced 500ms)
  const handleChangeText = (newText: string) => {
    setText(newText);
    if (draftTimerRef.current) clearTimeout(draftTimerRef.current);
    if (sessionId) {
      draftTimerRef.current = setTimeout(() => {
        saveDraft(sessionId, newText);
      }, 500);
    }
  };

  const handleSend = async () => {
    const trimmed = text.trim();
    if (!trimmed || disabled) return;
    await triggerHaptic('medium');
    onSend(trimmed);
    setText('');
    if (sessionId) clearDraft(sessionId);
    Keyboard.dismiss();
  };

  const handlePromptSelect = (prompt: string) => {
    setText(prompt);
    // Focus input so user can fill in {placeholders}
    setTimeout(() => inputRef.current?.focus(), 100);
  };

  const handlePlus = async () => {
    await triggerHaptic('light');
    try {
      const file = await pickDocument();
      if (file) {
        if (isTextFile(file) && onFile) {
          const FileSystem = require('expo-file-system');
          const content = await FileSystem.readAsStringAsync(file.uri);
          onFile(content, file.name);
        } else if (file.mimeType.startsWith('image/') && onImage) {
          onImage();
        } else {
          toast.warn(`${file.name} (${formatFileSize(file.size)}) — text files only for now`);
        }
      }
    } catch (e: any) {
      toast.error(e.message || 'Failed to pick file');
    }
  };

  const hasText = text.trim().length > 0;

  return (
    <View style={styles.container}>
      <View style={styles.inputRow}>
        <TouchableOpacity
          style={styles.iconBtn}
          onPress={handlePlus}
          disabled={disabled}>
          <Text style={styles.iconText}>+</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.iconBtn}
          onPress={() => {
            triggerHaptic('light');
            setShowPrompts(true);
          }}
          disabled={disabled}>
          <Text style={styles.iconText}>⚡</Text>
        </TouchableOpacity>

        <TextInput
          ref={inputRef}
          style={styles.input}
          value={text}
          onChangeText={handleChangeText}
          placeholder="Type a message..."
          placeholderTextColor={colors.textMuted}
          multiline
          maxLength={4000}
          editable={!disabled}
          returnKeyType="send"
          blurOnSubmit={false}
          onSubmitEditing={handleSend}
        />

        {editText && onCancelEdit && (
          <TouchableOpacity
            style={styles.cancelEditBtn}
            onPress={() => {
              setText('');
              onCancelEdit();
            }}>
            <Text style={styles.cancelEditText}>✕</Text>
          </TouchableOpacity>
        )}

        {disabled ? (
          <TouchableOpacity style={styles.stopBtn} onPress={onStop}>
            <Text style={styles.stopIcon}>■</Text>
          </TouchableOpacity>
        ) : hasText ? (
          <TouchableOpacity
            style={styles.sendBtn}
            onPress={handleSend}>
            <Text style={styles.sendIcon}>↑</Text>
          </TouchableOpacity>
        ) : (
          <View style={styles.voiceRow}>
            {onVoiceMessageToggle && !isRecording && (
              <TouchableOpacity
                style={styles.vmToggle}
                onPress={onVoiceMessageToggle}
                activeOpacity={0.7}>
                <Text style={styles.vmToggleText}>
                  {voiceMessageMode ? '🎤' : '🎙'}
                </Text>
              </TouchableOpacity>
            )}
            <TouchableOpacity
              style={[styles.voiceBtn, isRecording && styles.voiceBtnActive]}
              onPress={onVoice}>
              <Text style={styles.voiceIcon}>{isRecording ? '■' : voiceMessageMode ? '🎵' : '🎤'}</Text>
            </TouchableOpacity>
          </View>
        )}
      </View>

      {isRecording && (
        <View style={styles.recordingIndicator}>
          <View style={styles.recordingDot} />
          <Text style={styles.recordingText}>Recording...</Text>
        </View>
      )}

      <QuickPromptPicker
        visible={showPrompts}
        onClose={() => setShowPrompts(false)}
        onSelect={handlePromptSelect}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
    backgroundColor: colors.background,
    borderTopWidth: 1,
    borderTopColor: colors.border,
  },
  inputRow: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    gap: spacing.sm,
  },
  iconBtn: {
    width: 36,
    height: 36,
    borderRadius: radii.full,
    backgroundColor: colors.surface,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: colors.border,
  },
  iconText: {
    fontSize: 18,
    color: colors.textSecondary,
  },
  input: {
    flex: 1,
    ...typography.body,
    color: colors.text,
    backgroundColor: colors.surface,
    borderRadius: radii.xl,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
    maxHeight: 120,
    minHeight: 44,
  },
  sendBtn: {
    width: 40,
    height: 40,
    borderRadius: radii.full,
    backgroundColor: colors.primary,
    alignItems: 'center',
    justifyContent: 'center',
  },
  sendIcon: {
    color: colors.white,
    fontSize: 20,
    fontWeight: '700',
  },
  stopBtn: {
    width: 40,
    height: 40,
    borderRadius: radii.full,
    backgroundColor: colors.error,
    alignItems: 'center',
    justifyContent: 'center',
  },
  stopIcon: {
    color: colors.white,
    fontSize: 16,
  },
  cancelEditBtn: {
    width: 32,
    height: 32,
    borderRadius: radii.full,
    backgroundColor: colors.surface,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: colors.border,
  },
  cancelEditText: {
    color: colors.textMuted,
    fontSize: 14,
    fontWeight: '600',
  },
  voiceRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
  },
  vmToggle: {
    width: 28,
    height: 28,
    borderRadius: radii.full,
    backgroundColor: colors.surface,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: colors.border,
  },
  vmToggleText: {
    fontSize: 12,
  },
  voiceBtn: {
    width: 40,
    height: 40,
    borderRadius: radii.full,
    backgroundColor: colors.surface,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: colors.border,
  },
  voiceBtnActive: {
    backgroundColor: colors.error + '20',
    borderColor: colors.error,
  },
  voiceIcon: {
    fontSize: 18,
  },
  recordingIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.xs,
    marginTop: spacing.xs,
  },
  recordingDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: colors.error,
  },
  recordingText: {
    ...typography.small,
    color: colors.error,
    fontWeight: '600',
  },
});
