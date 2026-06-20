import React, {useState} from 'react';
import {
  View,
  TouchableOpacity,
  StyleSheet,
} from 'react-native';
import {Markdown} from './Markdown';
import {colors, spacing, radii} from '../theme';
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
        {isUser ? (
          <Markdown content={message.content} style={styles.userText} />
        ) : (
          <Markdown content={message.content || 'Thinking...'} style={styles.assistantText} />
        )}
      </TouchableOpacity>

      {showActions && !isUser && (
        <View style={styles.actions}>
          <TouchableOpacity
            style={styles.actionBtn}
            onPress={() => { onFeedback?.(true); setShowActions(false); }}>
            <View style={[styles.actionIcon, {backgroundColor: colors.success + '20'}]}>
              <Markdown content="👍" />
            </View>
          </TouchableOpacity>
          <TouchableOpacity
            style={styles.actionBtn}
            onPress={() => { onFeedback?.(false); setShowActions(false); }}>
            <View style={[styles.actionIcon, {backgroundColor: colors.error + '20'}]}>
              <Markdown content="👎" />
            </View>
          </TouchableOpacity>
          <TouchableOpacity
            style={styles.actionBtn}
            onPress={() => { onRegenerate?.(); setShowActions(false); }}>
            <View style={[styles.actionIcon, {backgroundColor: colors.primary + '20'}]}>
              <Markdown content="↻" />
            </View>
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
  actionBtn: {},
  actionIcon: {
    width: 32,
    height: 32,
    borderRadius: radii.full,
    alignItems: 'center',
    justifyContent: 'center',
  },
});
