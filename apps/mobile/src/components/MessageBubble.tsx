import React, {useState} from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Alert,
} from 'react-native';
import {colors, spacing, radii, typography} from '../theme';
import type {Message} from '../types';

interface Props {
  message: Message;
  onRegenerate?: () => void;
  onFeedback?: (positive: boolean) => void;
}

export function MessageBubble({message, onRegenerate, onFeedback}: Props) {
  const isUser = message.role === 'user';
  const [showActions, setShowActions] = useState(false);

  const handleLongPress = () => {
    if (isUser) return;
    setShowActions(!showActions);
  };

  return (
    <View style={[styles.row, isUser && styles.rowUser]}>
      <TouchableOpacity
        style={[styles.bubble, isUser ? styles.userBubble : styles.assistantBubble]}
        onLongPress={handleLongPress}
        activeOpacity={0.8}>
        <Text
          style={[styles.text, isUser ? styles.userText : styles.assistantText]}
          selectable>
          {message.content || (isUser ? '' : 'Thinking...')}
        </Text>
      </TouchableOpacity>

      {showActions && !isUser && (
        <View style={styles.actions}>
          <TouchableOpacity
            style={styles.actionBtn}
            onPress={() => {
              onFeedback?.(true);
              setShowActions(false);
            }}>
            <Text style={styles.actionText}>👍</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={styles.actionBtn}
            onPress={() => {
              onFeedback?.(false);
              setShowActions(false);
            }}>
            <Text style={styles.actionText}>👎</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={styles.actionBtn}
            onPress={() => {
              onRegenerate?.();
              setShowActions(false);
            }}>
            <Text style={styles.actionText}>↻</Text>
          </TouchableOpacity>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  row: {
    paddingHorizontal: spacing.lg,
    marginBottom: spacing.sm,
    alignItems: 'flex-start',
  },
  rowUser: {
    alignItems: 'flex-end',
  },
  bubble: {
    maxWidth: '80%',
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm + 2,
    borderRadius: radii.lg,
  },
  userBubble: {
    backgroundColor: colors.primary,
    borderBottomRightRadius: radii.sm,
  },
  assistantBubble: {
    backgroundColor: colors.surface,
    borderBottomLeftRadius: radii.sm,
  },
  text: {
    ...typography.body,
  },
  userText: {
    color: colors.white,
  },
  assistantText: {
    color: colors.text,
  },
  actions: {
    flexDirection: 'row',
    marginTop: spacing.xs,
    gap: spacing.xs,
  },
  actionBtn: {
    width: 32,
    height: 32,
    borderRadius: radii.full,
    backgroundColor: colors.surface,
    alignItems: 'center',
    justifyContent: 'center',
  },
  actionText: {
    fontSize: 14,
  },
});
