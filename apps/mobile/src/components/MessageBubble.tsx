import React, {useState, useRef} from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Alert,
  Animated,
  PanResponder,
} from 'react-native';
import {Markdown} from './Markdown';
import {copyToClipboard} from '../services/clipboard';
import {triggerHaptic} from '../services/haptics';
import {colors, spacing, radii, typography} from '../theme';
import type {Message} from '../types';

function formatTime(ts: number): string {
  const d = new Date(ts);
  const h = d.getHours();
  const m = d.getMinutes().toString().padStart(2, '0');
  const ampm = h >= 12 ? 'PM' : 'AM';
  const h12 = h % 12 || 12;
  return `${h12}:${m} ${ampm}`;
}

const SWIPE_THRESHOLD = -80;
const DELETE_WIDTH = 72;

interface Props {
  message: Message;
  highlight?: boolean;
  onRegenerate?: () => void;
  onFeedback?: (positive: boolean) => void;
  onDelete?: () => void;
}

export function MessageBubble({message, highlight, onRegenerate, onFeedback, onDelete}: Props) {
  const isUser = message.role === 'user';
  const [showActions, setShowActions] = useState(false);
  const translateX = useRef(new Animated.Value(0)).current;
  const isSwipeOpen = useRef(false);

  const panResponder = useRef(
    PanResponder.create({
      onMoveShouldSetPanResponder: (_, gestureState) => {
        return Math.abs(gestureState.dx) > 10 && Math.abs(gestureState.dx) > Math.abs(gestureState.dy);
      },
      onPanResponderMove: (_, gestureState) => {
        if (gestureState.dx < 0) {
          const clamped = Math.max(gestureState.dx, -DELETE_WIDTH - 20);
          translateX.setValue(clamped);
        } else if (isSwipeOpen.current) {
          translateX.setValue(gestureState.dx - DELETE_WIDTH);
        }
      },
      onPanResponderRelease: (_, gestureState) => {
        if (gestureState.dx < SWIPE_THRESHOLD && !isSwipeOpen.current) {
          // Swipe left → reveal delete
          Animated.spring(translateX, {
            toValue: -DELETE_WIDTH,
            useNativeDriver: true,
          }).start();
          isSwipeOpen.current = true;
          triggerHaptic('light');
        } else {
          // Snap back
          Animated.spring(translateX, {
            toValue: 0,
            useNativeDriver: true,
          }).start();
          isSwipeOpen.current = false;
        }
      },
    }),
  ).current;

  const handleLongPress = () => {
    setShowActions(!showActions);
  };

  const handleCopy = async () => {
    const ok = await copyToClipboard(message.content);
    setShowActions(false);
    if (ok) {
      Alert.alert('Copied', 'Message copied to clipboard');
    }
  };

  const handleDelete = () => {
    triggerHaptic('medium');
    Alert.alert('Delete message', 'Remove this message from the conversation?', [
      {text: 'Cancel', style: 'cancel'},
      {
        text: 'Delete',
        style: 'destructive',
        onPress: () => {
          Animated.timing(translateX, {
            toValue: 0,
            duration: 200,
            useNativeDriver: true,
          }).start(() => {
            isSwipeOpen.current = false;
            onDelete?.();
          });
        },
      },
    ]);
  };

  return (
    <View style={[styles.row, isUser && styles.rowUser]}>
      {/* Delete button behind the bubble */}
      <View style={styles.deleteContainer}>
        <TouchableOpacity style={styles.deleteBtn} onPress={handleDelete} activeOpacity={0.7}>
          <Text style={styles.deleteIcon}>🗑</Text>
        </TouchableOpacity>
      </View>

      {/* Swipeable bubble */}
      <Animated.View
        style={[styles.swipeable, {transform: [{translateX}]}]}
        {...panResponder.panHandlers}>
        <TouchableOpacity
          style={[styles.bubble, isUser ? styles.userBubble : styles.assistantBubble, highlight && styles.highlight]}
          onLongPress={handleLongPress}
          activeOpacity={0.8}>
          {isUser ? (
            <Markdown content={message.content} style={styles.userText} />
          ) : (
            <Markdown content={message.content || 'Thinking...'} style={styles.assistantText} />
          )}
        </TouchableOpacity>
      </Animated.View>

      <Text style={[styles.timestamp, isUser && styles.timestampUser]}>
        {formatTime(message.timestamp)}
      </Text>

      {showActions && (
        <View style={styles.actions}>
          <TouchableOpacity style={styles.actionBtn} onPress={handleCopy}>
            <View style={[styles.actionIcon, {backgroundColor: colors.surface}]}>
              <Markdown content="📋" />
            </View>
          </TouchableOpacity>
          {!isUser && (
            <>
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
            </>
          )}
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
    overflow: 'visible',
  },
  rowUser: {
    alignItems: 'flex-end',
  },
  deleteContainer: {
    position: 'absolute',
    right: spacing.lg,
    top: 0,
    bottom: 0,
    justifyContent: 'center',
    width: DELETE_WIDTH,
    alignItems: 'center',
  },
  deleteBtn: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: colors.error,
    alignItems: 'center',
    justifyContent: 'center',
  },
  deleteIcon: {
    fontSize: 16,
  },
  swipeable: {
    maxWidth: '80%',
  },
  bubble: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm + 2,
    borderRadius: radii.lg,
  },
  highlight: {
    borderWidth: 2,
    borderColor: colors.primary + '60',
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
  timestamp: {
    ...typography.small,
    color: colors.textMuted,
    marginTop: 2,
    paddingHorizontal: spacing.md,
  },
  timestampUser: {
    textAlign: 'right',
  },
});
