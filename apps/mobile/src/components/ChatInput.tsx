import React, {useState, useRef, useEffect} from 'react';
import {
  TextInput,
  Pressable,
  Keyboard,
  Text,
} from 'react-native';
import {YStack, XStack, useTheme} from 'tamagui';
import {triggerHaptic} from '../services/haptics';
import {QuickPromptPicker} from './QuickPromptPicker';
import {pickDocument, isTextFile, formatFileSize} from '../services/file-upload';
import {getDraft, saveDraft, clearDraft} from '../services/drafts';
import {toast} from '../services/toast';
import {Icon} from './Icon';

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
  voiceMessageMode?: boolean;
  onVoiceMessageToggle?: () => void;
}

export function ChatInput({onSend, onImage, onVoice, onFile, disabled, onStop, isRecording, sessionId, editText, onCancelEdit, voiceMessageMode, onVoiceMessageToggle}: Props) {
  const theme = useTheme();
  const [text, setText] = useState('');
  const [showPrompts, setShowPrompts] = useState(false);
  const inputRef = useRef<TextInput>(null);
  const draftTimerRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    if (editText) {
      setText(editText);
      inputRef.current?.focus();
    }
  }, [editText]);

  useEffect(() => {
    if (sessionId) {
      getDraft(sessionId).then(setText);
    }
  }, [sessionId]);

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
    setTimeout(() => inputRef.current?.focus(), 100);
  };

  const handlePlus = async () => {
    await triggerHaptic('light');
    try {
      const file = await pickDocument();
      if (file) {
        if (isTextFile(file) && onFile) {
          try {
            const RNFS = require('react-native-fs');
            const content = await RNFS.readFile(decodeURIComponent(file.uri.replace('file://', '')));
            onFile(content, file.name);
          } catch {
            toast.warn('File reading not available — install react-native-fs');
            return;
          }
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
  const bg = theme.background?.val || '#FFFFFF';
  const border = theme.borderColor?.val || '#E5E7EB';
  const textColor = theme.color?.val || '#111827';
  const textSecondary = theme.color11?.val || '#6B7280';
  const textMuted = theme.color10?.val || '#9CA3AF';
  const primary = theme.color9?.val || '#7C52C4';

  const iconBtnStyle = {
    width: 36,
    height: 36,
    borderRadius: 12,
    backgroundColor: 'rgba(124, 82, 196, 0.06)',
    alignItems: 'center' as const,
    justifyContent: 'center' as const,
    borderWidth: 0.5,
    borderColor: 'rgba(124, 82, 196, 0.12)',
  };

  const inputStyle = {
    flex: 1,
    fontSize: 15,
    fontWeight: '400' as const,
    color: textColor,
    backgroundColor: 'rgba(124, 82, 196, 0.04)',
    borderRadius: 14,
    paddingHorizontal: 16,
    paddingVertical: 10,
    maxHeight: 120,
    minHeight: 42,
    borderWidth: 0.5,
    borderColor: 'rgba(124, 82, 196, 0.1)',
  };

  const circleBtn = (size: number, color: string) => ({
    width: size,
    height: size,
    borderRadius: 9999,
    backgroundColor: color,
    alignItems: 'center' as const,
    justifyContent: 'center' as const,
    shadowColor: '#7C52C4',
    shadowOffset: {width: 0, height: 2} as const,
    shadowOpacity: 0.2,
    shadowRadius: 6,
    elevation: 3,
  });

  return (
    <YStack paddingHorizontal={16} paddingVertical={8} backgroundColor={bg} borderTopWidth={1} borderTopColor={border}>
      <XStack alignItems="flex-end" gap={8}>
        <Pressable onPress={handlePlus} disabled={disabled} style={iconBtnStyle}>
          <Icon name="plus" size={18} color={textSecondary} />
        </Pressable>

        <Pressable
          onPress={() => {
            triggerHaptic('light');
            setShowPrompts(true);
          }}
          disabled={disabled}
          style={iconBtnStyle}>
          <Icon name="zap" size={18} color={textSecondary} />
        </Pressable>

        <TextInput
          ref={inputRef}
          value={text}
          onChangeText={handleChangeText}
          placeholder="Type a message..."
          placeholderTextColor={textMuted}
          multiline
          maxLength={4000}
          editable={!disabled}
          returnKeyType="send"
          blurOnSubmit={false}
          onSubmitEditing={handleSend}
          style={inputStyle}
        />

        {editText && onCancelEdit && (
          <Pressable
            onPress={() => {
              setText('');
              onCancelEdit();
            }}
            style={{
              width: 32,
              height: 32,
              borderRadius: 9999,
              backgroundColor: bg,
              alignItems: 'center',
              justifyContent: 'center',
              borderWidth: 0.5,
              borderColor: border,
            }}>
            <Icon name="x" size={14} color={textMuted} />
          </Pressable>
        )}

        {disabled ? (
          <Pressable onPress={onStop} style={circleBtn(40, '#EF4444')}>
            <Icon name="stop-circle" size={16} color="white" />
          </Pressable>
        ) : hasText ? (
          <Pressable onPress={handleSend} style={circleBtn(40, primary)}>
            <Icon name="arrow-up" size={20} color="white" />
          </Pressable>
        ) : (
          <XStack alignItems="center" gap={4}>
            {onVoiceMessageToggle && !isRecording && (
              <Pressable
                onPress={onVoiceMessageToggle}
                style={{
                  width: 28,
                  height: 28,
                  borderRadius: 9999,
                  backgroundColor: bg,
                  alignItems: 'center',
                  justifyContent: 'center',
                  borderWidth: 0.5,
                  borderColor: border,
                }}>
                <Icon name="mic" size={12} color={textSecondary} />
              </Pressable>
            )}
            <Pressable
              onPress={onVoice}
              style={[
                {
                  width: 40,
                  height: 40,
                  borderRadius: 9999,
                  backgroundColor: bg,
                  alignItems: 'center',
                  justifyContent: 'center',
                  borderWidth: 0.5,
                  borderColor: border,
                },
                isRecording && {
                  backgroundColor: '#FDE8E8',
                  borderColor: '#EF4444',
                },
              ]}>
              <Icon name={isRecording ? 'stop-circle' : voiceMessageMode ? 'music' : 'mic'} size={18} color={textSecondary} />
            </Pressable>
          </XStack>
        )}
      </XStack>

      {isRecording && (
        <XStack alignItems="center" justifyContent="center" gap={4} marginTop={4}>
          <YStack width={8} height={8} borderRadius={4} backgroundColor="#EF4444" />
          <TextInput
            value="Recording..."
            editable={false}
            style={{
              fontSize: 11,
              fontWeight: '600',
              letterSpacing: 0.2,
              color: '#EF4444',
            }}
          />
        </XStack>
      )}

      <QuickPromptPicker
        visible={showPrompts}
        onClose={() => setShowPrompts(false)}
        onSelect={handlePromptSelect}
      />
    </YStack>
  );
}
