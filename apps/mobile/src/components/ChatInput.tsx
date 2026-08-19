import React, {useState, useRef, useEffect} from 'react';
import {
  TextInput,
  Pressable,
  Keyboard,
  Alert,
} from 'react-native';
import {YStack, XStack, Text} from 'tamagui';
import {useColors} from '../theme/colors';
import {triggerHaptic} from '../services/haptics';
import {QuickPromptPicker} from './QuickPromptPicker';
import {SlashCommandPicker} from './SlashCommandPicker';
import {pickDocument, isTextFile, formatFileSize} from '../services/file-upload';
import {takePhoto, pickImage, imageDataUrl} from '../services/image-upload';
import {getDraft, saveDraft, clearDraft} from '../services/drafts';
import {toast} from '../services/toast';
import {Icon} from './Icon';
import type {ChatCommand} from '../services/chat-commands';

interface Props {
  onSend: (text: string) => void;
  onSendWithImages?: (text: string, images: string[]) => void;
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
  onExecuteCommand?: (command: ChatCommand, args: string[]) => void;
}

export function ChatInput({onSend, onSendWithImages, onImage, onVoice, onFile, disabled, onStop, isRecording, sessionId, editText, onCancelEdit, voiceMessageMode, onVoiceMessageToggle, onExecuteCommand}: Props) {
  const colors = useColors();
  const [text, setText] = useState('');
  const [showPrompts, setShowPrompts] = useState(false);
  const [showSlash, setShowSlash] = useState(false);
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
    setShowSlash(newText.startsWith('/') && newText.length > 0);
    if (draftTimerRef.current) clearTimeout(draftTimerRef.current);
    if (sessionId) {
      draftTimerRef.current = setTimeout(() => {
        saveDraft(sessionId, newText);
      }, 500);
    }
  };

  const handleSlashSelect = (command: string) => {
    setText(command + ' ');
    setShowSlash(false);
    setTimeout(() => inputRef.current?.focus(), 100);
  };

  const handleSlashExecute = (command: ChatCommand, args: string[]) => {
    setShowSlash(false);
    setText('');
    if (sessionId) clearDraft(sessionId);
    if (onExecuteCommand) {
      onExecuteCommand(command, args);
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
    Alert.alert('Add Attachment', 'Choose an option', [
      {
        text: 'Camera',
        onPress: async () => {
          try {
            const photo = await takePhoto();
            if (photo) {
              const dataUrl = imageDataUrl(photo);
              if (onSendWithImages) {
                onSendWithImages('What do you see in this image?', [dataUrl]);
              } else if (onImage) {
                onImage();
              }
            }
          } catch (e: any) {
            toast.error(e.message || 'Failed to take photo');
          }
        },
      },
      {
        text: 'Gallery',
        onPress: async () => {
          if (onImage) {
            onImage();
          }
        },
      },
      {
        text: 'File',
        onPress: async () => {
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
        },
      },
      {text: 'Cancel', style: 'cancel'},
    ]);
  };

  const hasText = text.trim().length > 0;
  const bg = colors.background;
  const border = colors.border;
  const textColor = colors.text;
  const textSecondary = colors.textSecondary;
  const textMuted = colors.textMuted;
  const primary = colors.primary;

  const iconBtnStyle = {
    width: 36,
    height: 36,
    borderRadius: 12,
    backgroundColor: colors.primaryAlpha(0.06),
    alignItems: 'center' as const,
    justifyContent: 'center' as const,
    borderWidth: 0.5,
    borderColor: colors.primaryAlpha(0.12),
  };

  const inputStyle = {
    flex: 1,
    fontSize: 15,
    fontWeight: '400' as const,
    color: textColor,
    backgroundColor: colors.primaryAlpha(0.04),
    borderRadius: 14,
    paddingHorizontal: 16,
    paddingVertical: 10,
    maxHeight: 120,
    minHeight: 42,
    borderWidth: 0.5,
    borderColor: colors.primaryAlpha(0.1),
  };

  const circleBtn = (size: number, color: string) => ({
    width: size,
    height: size,
    borderRadius: 9999,
    backgroundColor: color,
    alignItems: 'center' as const,
    justifyContent: 'center' as const,
    shadowColor: primary,
    shadowOffset: {width: 0, height: 2} as const,
    shadowOpacity: 0.2,
    shadowRadius: 6,
    elevation: 3,
  });

  return (
    <YStack paddingHorizontal={16} paddingVertical={8} backgroundColor={bg} borderTopWidth={1} borderTopColor={border}>
      <XStack alignItems="flex-end" gap={8}>
        <Pressable onPress={handlePlus} disabled={disabled} style={iconBtnStyle} accessibilityLabel="Add attachment">
          <Icon name="plus" size={18} color={textSecondary} />
        </Pressable>

        <Pressable
          onPress={() => {
            triggerHaptic('light');
            setText('/');
            setShowSlash(true);
            setTimeout(() => inputRef.current?.focus(), 100);
          }}
          disabled={disabled}
          style={iconBtnStyle}
          accessible accessibilityRole="button" accessibilityLabel="Slash commands">
          <Text fontSize={16} fontWeight="600" color={textSecondary} fontFamily="JetBrainsMono-Regular">/</Text>
        </Pressable>

        <Pressable
          onPress={() => {
            triggerHaptic('light');
            setShowPrompts(true);
          }}
          disabled={disabled}
          style={iconBtnStyle}
          accessibilityLabel="Open quick prompts"
          accessibilityRole="button"
          accessible={true}>
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
            }}
            accessibilityLabel="Cancel edit"
            accessibilityRole="button"
            accessible={true}>
            <Icon name="x" size={14} color={textMuted} />
          </Pressable>
        )}

        {disabled ? (
          <Pressable onPress={onStop} style={circleBtn(40, colors.error)} accessibilityLabel="Stop" accessibilityRole="button" accessible={true}>
            <Icon name="stop-circle" size={16} color="white" />
          </Pressable>
        ) : hasText ? (
          <Pressable onPress={handleSend} style={circleBtn(40, primary)} accessibilityLabel="Send message" accessibilityRole="button" accessible={true}>
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
                }}
                accessibilityLabel="Toggle voice message"
                accessibilityRole="button"
                accessible={true}>
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
                  backgroundColor: colors.errorLight,
                  borderColor: colors.error,
                },
              ]}
              accessibilityLabel={isRecording ? 'Stop recording' : 'Start voice input'}
              accessibilityRole="button"
              accessible={true}>
              <Icon name={isRecording ? 'stop-circle' : voiceMessageMode ? 'music' : 'mic'} size={18} color={textSecondary} />
            </Pressable>
          </XStack>
        )}
      </XStack>

      {isRecording && (
        <XStack alignItems="center" justifyContent="center" gap={4} marginTop={4}>
          <YStack width={8} height={8} borderRadius={4} backgroundColor={colors.error} />
          <TextInput
            value="Recording..."
            editable={false}
            style={{
              fontSize: 11,
              fontWeight: '600',
              letterSpacing: 0.2,
              color: colors.error,
            }}
          />
        </XStack>
      )}

      <QuickPromptPicker
        visible={showPrompts}
        onClose={() => setShowPrompts(false)}
        onSelect={handlePromptSelect}
      />

      <SlashCommandPicker
        visible={showSlash}
        query={text}
        onSelect={handleSlashSelect}
        onExecute={handleSlashExecute}
        onClose={() => setShowSlash(false)}
      />
    </YStack>
  );
}
