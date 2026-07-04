import React, {useState, useRef} from 'react';
import {
  View,
  TextInput,
  TouchableOpacity,
  Text,
  StyleSheet,
  Keyboard,
  Alert,
} from 'react-native';
import {colors, spacing, radii, typography} from '../theme';
import {triggerHaptic} from '../services/haptics';

interface Props {
  onSend: (text: string) => void;
  onImage?: (base64: string) => void;
  onVoice?: () => void;
  disabled?: boolean;
  onStop?: () => void;
  isRecording?: boolean;
}

export function ChatInput({onSend, onImage, onVoice, disabled, onStop, isRecording}: Props) {
  const [text, setText] = useState('');
  const inputRef = useRef<TextInput>(null);

  const handleSend = async () => {
    const trimmed = text.trim();
    if (!trimmed || disabled) return;
    await triggerHaptic('medium');
    onSend(trimmed);
    setText('');
    Keyboard.dismiss();
  };

  const handleImage = async () => {
    await triggerHaptic('light');
    if (onImage) {
      onImage('');
    }
  };

  const handleVoice = async () => {
    await triggerHaptic('light');
    if (onVoice) {
      onVoice();
    }
  };

  const hasText = text.trim().length > 0;

  return (
    <View style={styles.container}>
      <View style={styles.inputRow}>
        <TouchableOpacity
          style={styles.iconBtn}
          onPress={handleImage}
          disabled={disabled}>
          <Text style={styles.iconText}>+</Text>
        </TouchableOpacity>

        <TextInput
          ref={inputRef}
          style={styles.input}
          value={text}
          onChangeText={setText}
          placeholder="Type a message..."
          placeholderTextColor={colors.textMuted}
          multiline
          maxLength={4000}
          editable={!disabled}
          returnKeyType="send"
          blurOnSubmit={false}
          onSubmitEditing={handleSend}
        />

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
          <TouchableOpacity
            style={[styles.voiceBtn, isRecording && styles.voiceBtnActive]}
            onPress={handleVoice}>
            <Text style={styles.voiceIcon}>{isRecording ? '■' : '🎤'}</Text>
          </TouchableOpacity>
        )}
      </View>

      {isRecording && (
        <View style={styles.recordingIndicator}>
          <View style={styles.recordingDot} />
          <Text style={styles.recordingText}>Recording...</Text>
        </View>
      )}
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
    fontSize: 20,
    color: colors.textSecondary,
    fontWeight: '600',
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
