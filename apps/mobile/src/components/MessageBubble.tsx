import React, {useState, useRef, useEffect, useMemo} from 'react';
import {
  View,
  Text,
  StyleSheet,
  Alert,
  Animated,
  PanResponder,
  Modal,
  Pressable,
  Image,
  Share,
} from 'react-native';
import {useTheme} from 'tamagui';
import {Markdown} from './Markdown';
import {copyToClipboard} from '../services/clipboard';
import {triggerHaptic} from '../services/haptics';
import {sounds} from '../services/sounds';
import {addBookmark, removeBookmark, isBookmarked} from '../services/bookmarks';
import {pinMessage, unpinMessage, isPinned} from '../services/pins';
import {getMessageReactions, toggleReaction, REACTION_EMOJIS, type ReactionEmoji} from '../services/reactions';
import {saveToKnowledge, getKnowledgeForMessage} from '../services/knowledge-store';
import {toast} from '../services/toast';
import {AudioPlayer} from './AudioPlayer';
import {Icon, type IconName} from './Icon';
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

// Spacing/radii/typography constants
const S = {xs: 4, sm: 8, md: 12, lg: 16, xl: 20};
const R = {sm: 6, md: 10, lg: 16, xl: 20, full: 9999};
const T = {body: {fontSize: 15, lineHeight: 22, letterSpacing: 0.1}, small: {fontSize: 11, lineHeight: 15, letterSpacing: 0.2}};
const COL = {
  white: '#FFFFFF',
  error: '#EF4444',
  overlay: 'rgba(0,0,0,0.4)',
  userBubble: '#7C52C4',
  assistantBubble: 'rgba(124, 82, 196, 0.06)',
  assistantBubbleBorder: 'rgba(124, 82, 196, 0.1)',
};

interface ContextAction {
  icon: IconName;
  label: string;
  destructive?: boolean;
  onPress: () => void;
}

interface Props {
  message: Message;
  sessionId?: string;
  highlight?: boolean;
  streaming?: boolean;
  onRegenerate?: () => void;
  onFeedback?: (positive: boolean) => void;
  onDelete?: () => void;
  onRetry?: () => void;
  onEdit?: (newContent: string) => void;
  onReply?: () => void;
  onForward?: () => void;
  selectMode?: boolean;
  selected?: boolean;
  onSelect?: () => void;
  onLongPressSelect?: () => void;
}

export function MessageBubble({message, sessionId, highlight, streaming, onRegenerate, onFeedback, onDelete, onRetry, onEdit, onReply, onForward, selectMode, selected, onSelect, onLongPressSelect}: Props) {
  const theme = useTheme();
  const bg = theme.background?.val || '#FFFFFF';
  const border = theme.borderColor?.val || '#E4E0F2';
  const textColor = theme.color?.val || '#1A1625';
  const textMuted = theme.color10?.val || '#827A96';
  const primary = theme.color9?.val || '#7C52C4';

  const styles = useMemo(() => StyleSheet.create({
    row: {
      paddingHorizontal: S.lg,
      marginBottom: 6,
      alignItems: 'flex-start',
      overflow: 'visible',
    },
    rowUser: {
      alignItems: 'flex-end',
    },
    deleteContainer: {
      position: 'absolute',
      right: S.lg,
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
      backgroundColor: COL.error,
      alignItems: 'center',
      justifyContent: 'center',
    },
    selectCheckbox: {
      width: 32,
      height: 32,
      justifyContent: 'center',
      alignItems: 'center',
      marginRight: S.xs,
    },
    checkbox: {
      width: 22,
      height: 22,
      borderRadius: 11,
      borderWidth: 2,
      borderColor: border,
      backgroundColor: bg,
      justifyContent: 'center',
      alignItems: 'center',
    },
    checkboxSelected: {
      backgroundColor: primary,
      borderColor: primary,
    },
    swipeable: {
      maxWidth: '80%',
    },
    bubble: {
      paddingHorizontal: S.md + 2,
      paddingVertical: 10,
      borderRadius: R.lg,
    },
    highlight: {
      borderWidth: 2,
      borderColor: primary + '60',
    },
    userBubble: {
      backgroundColor: COL.userBubble,
      borderBottomRightRadius: R.sm,
    },
    assistantBubble: {
      backgroundColor: COL.assistantBubble,
      borderWidth: 0.5,
      borderColor: COL.assistantBubbleBorder,
      borderBottomLeftRadius: R.sm,
    },
    pinnedBubble: {
      borderWidth: 1,
      borderColor: primary + '40',
    },
    userText: {
      color: COL.white,
    },
    assistantText: {
      color: textColor,
    },
    pinIcon: {
      marginBottom: 4,
    },
    imageContainer: {
      marginBottom: S.xs,
      gap: S.xs,
    },
    messageImage: {
      width: 200,
      height: 150,
      borderRadius: R.md,
    },
    timestamp: {
      ...T.small,
      color: textMuted,
      marginTop: 3,
    },
    readTime: {
      color: textMuted,
      opacity: 0.7,
    },
    timestampRow: {
      marginTop: 3,
      paddingHorizontal: 0,
    },
    timestampUser: {
      textAlign: 'right',
    },
    timestampRowUser: {
      alignItems: 'flex-end',
    },
    reactionRow: {
      flexDirection: 'row',
      gap: 4,
      marginTop: 4,
      paddingHorizontal: 0,
      flexWrap: 'wrap',
    },
    reactionRowUser: {
      justifyContent: 'flex-end',
    },
    reactionBadge: {
      backgroundColor: primary + '20',
      borderRadius: R.full,
      paddingHorizontal: 8,
      paddingVertical: 2,
    },
    reactionEmoji: {
      fontSize: 16,
    },
    reactionPicker: {
      flexDirection: 'row',
      gap: 4,
      marginTop: 6,
      paddingHorizontal: 0,
      flexWrap: 'wrap',
      alignItems: 'center',
    },
    reactionPickerUser: {
      justifyContent: 'flex-end',
    },
    reactionOption: {
      width: 36,
      height: 36,
      borderRadius: 18,
      backgroundColor: bg,
      alignItems: 'center',
      justifyContent: 'center',
      borderWidth: 1,
      borderColor: border,
    },
    reactionOptionActive: {
      backgroundColor: primary + '20',
      borderColor: primary,
    },
    reactionOptionText: {
      fontSize: 18,
    },
    reactionClose: {
      width: 28,
      height: 28,
      borderRadius: 14,
      backgroundColor: bg,
      alignItems: 'center',
      justifyContent: 'center',
      marginLeft: 4,
    },
    overlay: {
      flex: 1,
      backgroundColor: COL.overlay,
      justifyContent: 'center',
      alignItems: 'center',
      padding: 24,
    },
    contextMenu: {
      backgroundColor: bg,
      borderRadius: R.lg,
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
      borderBottomColor: border,
    },
    contextPreviewText: {
      ...T.small,
      color: textMuted,
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
      borderTopColor: border,
    },
    contextLabel: {
      ...T.body,
      color: textColor,
      fontSize: 15,
    },
    contextLabelDestructive: {
      color: COL.error,
    },
  }), [bg, border, textColor, textMuted, primary]);

  const isUser = message.role === 'user';
  const [showContextMenu, setShowContextMenu] = useState(false);
  const [bookmarked, setBookmarked] = useState(false);
  const [pinned, setPinned] = useState(false);
  const [reactions, setReactions] = useState<ReactionEmoji[]>([]);
  const [showReactionPicker, setShowReactionPicker] = useState(false);
  const [showFullDate, setShowFullDate] = useState(false);
  const [collapsed, setCollapsed] = useState(true);
  const [savedToKnowledge, setSavedToKnowledge] = useState(false);
  const translateX = useRef(new Animated.Value(0)).current;
  const opacity = useRef(new Animated.Value(0)).current;
  const translateY = useRef(new Animated.Value(8)).current;

  useEffect(() => {
    Animated.parallel([
      Animated.timing(opacity, {toValue: 1, duration: 200, useNativeDriver: true}),
      Animated.timing(translateY, {toValue: 0, duration: 200, useNativeDriver: true}),
    ]).start();
    isBookmarked(message.content, message.id).then(setBookmarked);
    if (sessionId) isPinned(sessionId, message.id).then(setPinned);
    getMessageReactions(message.id).then(setReactions);
    getKnowledgeForMessage(message.id).then(f => setSavedToKnowledge(!!f));
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

  const handleTogglePin = async () => {
    setShowContextMenu(false);
    if (!sessionId) return;
    if (pinned) {
      await unpinMessage(sessionId, message.id);
      setPinned(false);
      toast.info('Message unpinned');
    } else {
      await pinMessage(sessionId, message.id);
      setPinned(true);
      triggerHaptic('success');
      toast.success('Message pinned');
    }
  };

  const handleToggleReaction = async (emoji: ReactionEmoji) => {
    const updated = await toggleReaction(message.id, emoji);
    setReactions(updated);
    setShowReactionPicker(false);
    triggerHaptic('light');
  };

  const handleSaveToKnowledge = async () => {
    setShowContextMenu(false);
    const ok = await saveToKnowledge(message.content, message.role as 'user' | 'assistant', message.id);
    if (ok) {
      setSavedToKnowledge(true);
      triggerHaptic('success');
      toast.success('Saved to knowledge base');
    } else {
      toast.info('Already in knowledge base');
    }
  };

  const contextActions: ContextAction[] = [
    {icon: 'copy', label: 'Copy', onPress: handleCopy},
    {icon: 'external-link', label: 'Share', onPress: async () => {
      setShowContextMenu(false);
      await Share.share({message: message.content});
    }},
    ...(onReply ? [{icon: 'reply' as IconName, label: 'Reply', onPress: () => {
      setShowContextMenu(false);
      onReply();
    }}] : []),
    ...(onForward ? [{icon: 'forward' as IconName, label: 'Forward', onPress: () => {
      setShowContextMenu(false);
      onForward();
    }}] : []),
    ...(isUser && onEdit ? [{icon: 'edit' as IconName, label: 'Edit', onPress: () => {
      setShowContextMenu(false);
      onEdit(message.content);
    }}] : []),
    {icon: (pinned ? 'pin' as IconName : 'pin-off' as IconName), label: pinned ? 'Unpin' : 'Pin', onPress: handleTogglePin},
    {icon: (bookmarked ? 'star' as IconName : 'star-outline' as IconName), label: bookmarked ? 'Remove bookmark' : 'Bookmark', onPress: handleToggleBookmark},
    {icon: (savedToKnowledge ? 'check' as IconName : 'book-open' as IconName), label: savedToKnowledge ? 'In knowledge base' : 'Save to knowledge', onPress: handleSaveToKnowledge},
    {icon: 'smile-plus' as IconName, label: 'React', onPress: () => { setShowContextMenu(false); setShowReactionPicker(true); }},
    ...(isUser ? [] : [
      {icon: 'thumbs-up' as IconName, label: 'Good response', onPress: () => { setShowContextMenu(false); onFeedback?.(true); }},
      {icon: 'thumbs-down' as IconName, label: 'Bad response', onPress: () => { setShowContextMenu(false); onFeedback?.(false); }},
      {icon: 'refresh-cw' as IconName, label: 'Regenerate', onPress: () => { setShowContextMenu(false); onRegenerate?.(); }},
    ]),
    {icon: 'trash-2' as IconName, label: 'Delete', destructive: true, onPress: handleDelete},
  ];

  const lastTap = useRef<number>(0);

  const handleDoubleTap = () => {
    const now = Date.now();
    if (now - lastTap.current < 300) {
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
      {selectMode && (
        <Pressable
          style={styles.selectCheckbox}
          onPress={onSelect}>
          <View style={[styles.checkbox, selected && styles.checkboxSelected]}>
            {selected && <Icon name="check" size={12} color="white" />}
          </View>
        </Pressable>
      )}

      <View style={styles.deleteContainer}>
        <Pressable style={styles.deleteBtn} onPress={handleDelete}>
          <Icon name="trash-2" size={16} color="white" />
        </Pressable>
      </View>

      <Animated.View
        style={[styles.swipeable, {transform: [{translateX}]}]}
        {...panResponder.panHandlers}>
        <Pressable
          style={[styles.bubble, isUser ? styles.userBubble : styles.assistantBubble, highlight && styles.highlight, pinned && styles.pinnedBubble]}
          onPress={selectMode ? onSelect : handleDoubleTap}
          onLongPress={selectMode ? onSelect : (onLongPressSelect || handleLongPress)}>
          {pinned && (
            <View style={styles.pinIcon}>
              <Icon name="pin" size={12} color={primary} />
            </View>
          )}
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
          {(message._voice || message.audio_path || message.audio) && (
            <AudioPlayer
              audioUrl={message.audio}
              audioPath={message.audio_path}
              durationMs={message.audio_duration_ms}
            />
          )}
          {isUser ? (
            <Markdown content={message.content} />
          ) : (
            <Markdown
              content={collapsed && message.content.length > 500
                ? message.content.slice(0, 500) + '...'
                : message.content || 'Thinking...'}
              streaming={streaming}
            />
          )}
          {!isUser && !streaming && message.content && message.content.length > 500 && (
            <View style={{marginTop: 4}}>
              <Pressable onPress={() => setCollapsed(c => !c)} hitSlop={8}>
                <Text style={[styles.timestamp, {color: primary}]}>
                  {collapsed ? 'Show full response' : 'Show less'}
                </Text>
              </Pressable>
            </View>
          )}
        </Pressable>
      </Animated.View>

      <Pressable
        onPress={() => setShowFullDate(d => !d)}
        style={[styles.timestampRow, isUser && styles.timestampRowUser]}>
        <Text style={[styles.timestamp, isUser && styles.timestampUser]}>
          {showFullDate
            ? new Date(message.timestamp).toLocaleString()
            : formatTime(message.timestamp)}
          {message.content && message.content.length > 200 && (
            <Text style={styles.readTime}> · {Math.max(1, Math.ceil(message.content.split(/\s+/).length / 200))} min read</Text>
          )}
        </Text>
      </Pressable>

      {message.status === 'failed' && (
        <View style={{flexDirection: 'row', alignItems: 'center', gap: 6, marginTop: 4}}>
          <Icon name="triangle-alert" size={14} color={COL.error} />
          <Text style={[styles.timestamp, {color: COL.error}]}>Failed to send</Text>
          {onRetry && (
            <Pressable onPress={onRetry} hitSlop={8}>
              <View style={{flexDirection: 'row', alignItems: 'center', gap: 3}}>
                <Icon name="refresh-cw" size={12} color={primary} />
                <Text style={[styles.timestamp, {color: primary}]}>Retry</Text>
              </View>
            </Pressable>
          )}
        </View>
      )}

      {reactions.length > 0 && (
        <View style={[styles.reactionRow, isUser && styles.reactionRowUser]}>
          {reactions.map((emoji, i) => (
            <Pressable
              key={i}
              style={styles.reactionBadge}
              onPress={() => handleToggleReaction(emoji)}
              onLongPress={() => setShowReactionPicker(true)}>
              <Text style={styles.reactionEmoji}>{emoji}</Text>
            </Pressable>
          ))}
        </View>
      )}

      {showReactionPicker && (
        <View style={[styles.reactionPicker, isUser && styles.reactionPickerUser]}>
          {REACTION_EMOJIS.map(emoji => (
            <Pressable
              key={emoji}
              style={[styles.reactionOption, reactions.includes(emoji) && styles.reactionOptionActive]}
              onPress={() => handleToggleReaction(emoji)}>
              <Text style={styles.reactionOptionText}>{emoji}</Text>
            </Pressable>
          ))}
          <Pressable
            style={styles.reactionClose}
            onPress={() => setShowReactionPicker(false)}>
            <Icon name="x" size={14} color={textMuted} />
          </Pressable>
        </View>
      )}

      <Modal
        visible={showContextMenu}
        transparent
        animationType="fade"
        onRequestClose={() => setShowContextMenu(false)}>
        <Pressable style={styles.overlay} onPress={() => setShowContextMenu(false)}>
          <View style={[styles.contextMenu, isUser && styles.contextMenuUser]}>
            <View style={styles.contextPreview}>
              <Text style={styles.contextPreviewText} numberOfLines={2}>
                {message.content || 'Thinking...'}
              </Text>
            </View>

            {contextActions.map((action, i) => (
              <Pressable
                key={i}
                style={[styles.contextAction, action.destructive && styles.contextActionDestructive]}
                onPress={action.onPress}>
                <Icon name={action.icon} size={18} color={textColor} />
                <Text style={[styles.contextLabel, action.destructive && styles.contextLabelDestructive]}>
                  {action.label}
                </Text>
              </Pressable>
            ))}
          </View>
        </Pressable>
      </Modal>
    </Animated.View>
  );
}
