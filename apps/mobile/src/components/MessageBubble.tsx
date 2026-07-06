import React, {useState, useRef, useEffect} from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Alert,
  Animated,
  PanResponder,
  Modal,
  Pressable,
  Image,
  Share,
} from 'react-native';
import {Markdown} from './Markdown';
import {copyToClipboard} from '../services/clipboard';
import {triggerHaptic} from '../services/haptics';
import {sounds} from '../services/sounds';
import {addBookmark, removeBookmark, isBookmarked} from '../services/bookmarks';
import {toast} from '../services/toast';
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

interface ContextAction {
  icon: string;
  label: string;
  destructive?: boolean;
  onPress: () => void;
}

interface Props {
  message: Message;
  highlight?: boolean;
  onRegenerate?: () => void;
  onFeedback?: (positive: boolean) => void;
  onDelete?: () => void;
  onRetry?: () => void;
}

export function MessageBubble({message, highlight, onRegenerate, onFeedback, onDelete, onRetry}: Props) {
  const isUser = message.role === 'user';
  const [showContextMenu, setShowContextMenu] = useState(false);
  const [bookmarked, setBookmarked] = useState(false);
  const translateX = useRef(new Animated.Value(0)).current;
  const opacity = useRef(new Animated.Value(0)).current;
  const translateY = useRef(new Animated.Value(8)).current;

  useEffect(() => {
    Animated.parallel([
      Animated.timing(opacity, {toValue: 1, duration: 200, useNativeDriver: true}),
      Animated.timing(translateY, {toValue: 0, duration: 200, useNativeDriver: true}),
    ]).start();
    // Check bookmark state
    isBookmarked(message.content, message.id).then(setBookmarked);
  }, []);
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
          Animated.spring(translateX, {
            toValue: -DELETE_WIDTH,
            useNativeDriver: true,
          }).start();
          isSwipeOpen.current = true;
          triggerHaptic('light');
        } else {
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
    triggerHaptic('medium');
    setShowContextMenu(true);
  };

  const handleCopy = async () => {
    setShowContextMenu(false);
    const ok = await copyToClipboard(message.content);
    if (ok) {
      triggerHaptic('success');
    }
  };

  const handleDelete = () => {
    setShowContextMenu(false);
    Alert.alert('Delete message', 'Remove this message from the conversation?', [
      {text: 'Cancel', style: 'cancel'},
      {
        text: 'Delete',
        style: 'destructive',
        onPress: () => {
          Animated.timing(translateX, {toValue: 0, duration: 200, useNativeDriver: true}).start(() => {
            isSwipeOpen.current = false;
            onDelete?.();
          });
        },
      },
    ]);
  };

  const handleToggleBookmark = async () => {
    setShowContextMenu(false);
    if (bookmarked) {
      await removeBookmark(message.id);
      setBookmarked(false);
      toast.info('Bookmark removed');
    } else {
      await addBookmark(message.content, message.role as 'user' | 'assistant', message.id);
      setBookmarked(true);
      triggerHaptic('success');
      toast.success('Message bookmarked');
    }
  };

  const contextActions: ContextAction[] = [
    {icon: '📋', label: 'Copy', onPress: handleCopy},
    {icon: '↗', label: 'Share', onPress: async () => {
      setShowContextMenu(false);
      await Share.share({message: message.content});
    }},
    {icon: bookmarked ? '★' : '☆', label: bookmarked ? 'Remove bookmark' : 'Bookmark', onPress: handleToggleBookmark},
    ...(isUser ? [] : [
      {icon: '👍', label: 'Good response', onPress: () => { setShowContextMenu(false); onFeedback?.(true); }},
      {icon: '👎', label: 'Bad response', onPress: () => { setShowContextMenu(false); onFeedback?.(false); }},
      {icon: '↻', label: 'Regenerate', onPress: () => { setShowContextMenu(false); onRegenerate?.(); }},
    ]),
    {icon: '🗑', label: 'Delete', destructive: true, onPress: handleDelete},
  ];

  const lastTap = useRef<number>(0);

  const handleDoubleTap = () => {
    const now = Date.now();
    if (now - lastTap.current < 300) {
      // Double tap — copy
      copyToClipboard(message.content).then(ok => {
        if (ok) {
          triggerHaptic('success');
          toast.success('Copied to clipboard');
        }
      });
    }
    lastTap.current = now;
  };

  return (
    <Animated.View style={[styles.row, isUser && styles.rowUser, {opacity, transform: [{translateY}]}]}>
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
          onPress={handleDoubleTap}
          onLongPress={handleLongPress}
          activeOpacity={0.8}>
          {message.images && message.images.length > 0 && (
            <View style={styles.imageContainer}>
              {message.images.map((uri, i) => (
                <Image
                  key={i}
                  source={{uri}}
                  style={styles.messageImage}
                  resizeMode="cover"
                />
              ))}
            </View>
          )}
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

      {/* Context Menu Modal */}
      <Modal
        visible={showContextMenu}
        transparent
        animationType="fade"
        onRequestClose={() => setShowContextMenu(false)}>
        <Pressable style={styles.overlay} onPress={() => setShowContextMenu(false)}>
          <View style={[styles.contextMenu, isUser && styles.contextMenuUser]}>
            {/* Preview */}
            <View style={styles.contextPreview}>
              <Text style={styles.contextPreviewText} numberOfLines={2}>
                {message.content || 'Thinking...'}
              </Text>
            </View>

            {/* Actions */}
            {contextActions.map((action, i) => (
              <TouchableOpacity
                key={i}
                style={[styles.contextAction, action.destructive && styles.contextActionDestructive]}
                onPress={action.onPress}
                activeOpacity={0.6}>
                <Text style={styles.contextIcon}>{action.icon}</Text>
                <Text style={[styles.contextLabel, action.destructive && styles.contextLabelDestructive]}>
                  {action.label}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        </Pressable>
      </Modal>
    </Animated.View>
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
  imageContainer: {
    marginBottom: spacing.xs,
    gap: spacing.xs,
  },
  messageImage: {
    width: 200,
    height: 150,
    borderRadius: radii.md,
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
  // Context menu
  overlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.4)',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 24,
  },
  contextMenu: {
    backgroundColor: colors.surface,
    borderRadius: radii.lg,
    width: '100%',
    maxWidth: 280,
    overflow: 'hidden',
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 8},
    shadowOpacity: 0.2,
    shadowRadius: 16,
    elevation: 8,
  },
  contextMenuUser: {
    alignItems: 'flex-end',
  },
  contextPreview: {
    paddingHorizontal: 16,
    paddingTop: 14,
    paddingBottom: 10,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  contextPreviewText: {
    ...typography.small,
    color: colors.textMuted,
    lineHeight: 18,
  },
  contextAction: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 13,
    gap: 12,
  },
  contextActionDestructive: {
    borderTopWidth: 1,
    borderTopColor: colors.border,
  },
  contextIcon: {
    fontSize: 18,
    width: 28,
    textAlign: 'center',
  },
  contextLabel: {
    ...typography.body,
    color: colors.text,
    fontSize: 15,
  },
  contextLabelDestructive: {
    color: colors.error,
  },
});
