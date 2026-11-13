import React, { useState, useRef, useEffect, useMemo } from 'react'
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
  TextInput,
} from 'react-native'
import { Markdown } from './Markdown'
import { useColors } from '../theme/colors'
import { S, R, T } from '../theme/scales'
import { copyToClipboard } from '../services/clipboard'
import { triggerHaptic } from '../services/haptics'
import { sounds } from '../services/sounds'
import { addBookmark, removeBookmark, isBookmarked } from '../services/bookmarks'
import { pinMessage, unpinMessage, isPinned } from '../services/pins'
import {
  getMessageReactions,
  toggleReaction,
  REACTION_EMOJIS,
  type ReactionEmoji,
} from '../services/reactions'
import { saveToKnowledge, getKnowledgeForMessage } from '../services/knowledge-store'
import { getMessageNotes, saveMessageNote, removeMessageNote } from '../services/message-notes'
import { toast } from '../services/toast'
import { AudioPlayer } from './AudioPlayer'
import { Icon, type IconName } from './Icon'
import type { Message } from '../types'

function formatTime(ts: number): string {
  const d = new Date(ts)
  const h = d.getHours()
  const m = d.getMinutes().toString().padStart(2, '0')
  const ampm = h >= 12 ? 'PM' : 'AM'
  const h12 = h % 12 || 12
  return `${h12}:${m} ${ampm}`
}

const SWIPE_THRESHOLD = -80
const DELETE_WIDTH = 72

interface ContextAction {
  icon: IconName
  label: string
  destructive?: boolean
  onPress: () => void
}

interface Props {
  message: Message
  sessionId?: string
  highlight?: boolean
  searchQuery?: string
  streaming?: boolean
  onRegenerate?: () => void
  onFeedback?: (positive: boolean) => void
  onDelete?: () => void
  onRetry?: () => void
  onEdit?: (newContent: string) => void
  onReply?: () => void
  onForward?: () => void
  onBranch?: () => void
  selectMode?: boolean
  selected?: boolean
  onSelect?: () => void
  onLongPressSelect?: () => void
}

export function MessageBubble({
  message,
  sessionId,
  highlight,
  searchQuery,
  streaming,
  onRegenerate,
  onFeedback,
  onDelete,
  onRetry,
  onEdit,
  onReply,
  onForward,
  onBranch,
  selectMode,
  selected,
  onSelect,
  onLongPressSelect,
}: Props) {
  const colors = useColors()
  const bg = colors.background
  const border = colors.border
  const textColor = colors.text
  const textMuted = colors.textMuted
  const primary = colors.primary

  const styles = useMemo(
    () =>
      StyleSheet.create({
        row: {
          paddingHorizontal: S.lg,
          marginBottom: 6,
          alignItems: 'stretch',
        },
        innerRow: {
          flexDirection: 'row',
          alignItems: 'stretch',
        },
        innerRowUser: {
          flexDirection: 'row-reverse',
        },
        deleteBtn: {
          position: 'absolute',
          right: 36,
          top: 0,
          bottom: 0,
          width: 36,
          justifyContent: 'center',
          alignItems: 'center',
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
          flex: 1,
          overflow: 'hidden',
        },
        bubble: {
          paddingHorizontal: S.md + 2,
          paddingVertical: 10,
          borderRadius: R.xl,
        },
        highlight: {
          borderWidth: 2,
          borderColor: primary + '60',
        },
        userBubble: {
          backgroundColor: colors.primary,
          borderBottomRightRadius: R.sm,
        },
        assistantBubble: {
          backgroundColor: colors.primaryAlpha(0.04),
          borderRadius: 4,
          paddingHorizontal: S.sm,
          paddingVertical: 2,
        },
        pinnedBubble: {
          borderWidth: 1,
          borderColor: primary + '40',
        },
        userText: {
          color: colors.white,
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
          ...T.badge,
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
          backgroundColor: colors.overlay(0.4),
          justifyContent: 'center',
          alignItems: 'center',
          padding: 24,
        },
        contextMenu: {
          backgroundColor: bg,
          borderRadius: R.xl,
          width: '100%',
          maxWidth: 280,
          overflow: 'hidden',
          shadowColor: '#000',
          shadowOffset: { width: 0, height: 8 },
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
          ...T.badge,
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
        },
        contextLabelDestructive: {
          color: colors.error,
        },
        noteStrip: {
          flexDirection: 'row',
          alignItems: 'flex-start',
          gap: 6,
          marginTop: 8,
          paddingVertical: 6,
          paddingHorizontal: 8,
          borderRadius: R.sm,
          backgroundColor: colors.primaryAlpha(0.08),
          borderWidth: 1,
          borderColor: colors.primaryAlpha(0.25),
        },
        noteText: {
          ...T.badge,
          color: textColor,
          flex: 1,
          fontStyle: 'italic',
        },
        noteEditor: {
          marginTop: 8,
          padding: 8,
          borderRadius: R.sm,
          borderWidth: 1,
          borderColor: border,
          backgroundColor: colors.whiteAlpha(0.06),
        },
        noteInput: {
          ...T.badge,
          color: textColor,
          minHeight: 44,
          textAlignVertical: 'top',
          padding: 0,
        },
        noteActions: {
          flexDirection: 'row',
          alignItems: 'center',
          gap: 10,
          marginTop: 8,
        },
        noteSaveBtn: {
          backgroundColor: primary,
          paddingHorizontal: 10,
          paddingVertical: 5,
          borderRadius: R.sm,
        },
        noteSaveLabel: {
          ...T.badge,
          color: '#FFFFFF',
          fontWeight: '600',
        },
      }),
    [bg, border, textColor, textMuted, primary],
  )

  const isUser = message.role === 'user'
  const [showContextMenu, setShowContextMenu] = useState(false)
  const [bookmarked, setBookmarked] = useState(false)
  const [pinned, setPinned] = useState(false)
  const [reactions, setReactions] = useState<ReactionEmoji[]>([])
  const [showReactionPicker, setShowReactionPicker] = useState(false)
  const [showFullDate, setShowFullDate] = useState(false)
  const [collapsed, setCollapsed] = useState(true)
  const [savedToKnowledge, setSavedToKnowledge] = useState(false)
  const [note, setNote] = useState<string | null>(null)
  const [noteEditorOpen, setNoteEditorOpen] = useState(false)
  const [noteDraft, setNoteDraft] = useState('')
  const translateX = useRef(new Animated.Value(0)).current
  const opacity = useRef(new Animated.Value(0)).current
  const translateY = useRef(new Animated.Value(8)).current

  useEffect(() => {
    Animated.parallel([
      Animated.timing(opacity, { toValue: 1, duration: 200, useNativeDriver: true }),
      Animated.timing(translateY, { toValue: 0, duration: 200, useNativeDriver: true }),
    ]).start()
    isBookmarked(message.content, message.id).then(setBookmarked)
    if (sessionId) isPinned(sessionId, message.id).then(setPinned)
    getMessageReactions(message.id).then(setReactions)
    getKnowledgeForMessage(message.id).then((f) => setSavedToKnowledge(!!f))
    if (sessionId && message.id) {
      getMessageNotes(sessionId)
        .then((notes) => setNote(notes.find((n) => n.messageId === message.id)?.content ?? null))
        .catch(() => {})
    }
  }, [])
  const isSwipeOpen = useRef(false)

  const panResponder = useRef(
    PanResponder.create({
      onMoveShouldSetPanResponder: (_, gestureState) => {
        return (
          Math.abs(gestureState.dx) > 10 && Math.abs(gestureState.dx) > Math.abs(gestureState.dy)
        )
      },
      onPanResponderMove: (_, gestureState) => {
        if (gestureState.dx < 0) {
          const clamped = Math.max(gestureState.dx, -DELETE_WIDTH - 20)
          translateX.setValue(clamped)
        } else if (isSwipeOpen.current) {
          translateX.setValue(gestureState.dx - DELETE_WIDTH)
        }
      },
      onPanResponderRelease: (_, gestureState) => {
        if (gestureState.dx < SWIPE_THRESHOLD && !isSwipeOpen.current) {
          Animated.spring(translateX, {
            toValue: -DELETE_WIDTH,
            useNativeDriver: true,
          }).start()
          isSwipeOpen.current = true
          triggerHaptic('light')
        } else {
          Animated.spring(translateX, {
            toValue: 0,
            useNativeDriver: true,
          }).start()
          isSwipeOpen.current = false
        }
      },
    }),
  ).current

  const handleLongPress = () => {
    triggerHaptic('medium')
    setShowContextMenu(true)
  }

  const handleCopy = async () => {
    setShowContextMenu(false)
    const ok = await copyToClipboard(message.content)
    if (ok) {
      triggerHaptic('success')
    }
  }

  const handleDelete = () => {
    setShowContextMenu(false)
    Alert.alert('Delete message', 'Remove this message from the conversation?', [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Delete',
        style: 'destructive',
        onPress: () => {
          Animated.timing(translateX, { toValue: 0, duration: 200, useNativeDriver: true }).start(
            () => {
              isSwipeOpen.current = false
              onDelete?.()
            },
          )
        },
      },
    ])
  }

  const handleToggleBookmark = async () => {
    setShowContextMenu(false)
    if (bookmarked) {
      await removeBookmark(message.id)
      setBookmarked(false)
      toast.info('Bookmark removed')
    } else {
      await addBookmark(message.content, message.role as 'user' | 'assistant', message.id)
      setBookmarked(true)
      triggerHaptic('success')
      toast.success('Message bookmarked')
    }
  }

  const handleTogglePin = async () => {
    setShowContextMenu(false)
    if (!sessionId) return
    if (pinned) {
      await unpinMessage(sessionId, message.id)
      setPinned(false)
      toast.info('Message unpinned')
    } else {
      await pinMessage(sessionId, message.id)
      setPinned(true)
      triggerHaptic('success')
      toast.success('Message pinned')
    }
  }

  const handleOpenNoteEditor = () => {
    setShowContextMenu(false)
    setNoteDraft(note || '')
    setNoteEditorOpen(true)
  }

  const handleSaveNote = async () => {
    if (!sessionId || !message.id) return
    const trimmed = noteDraft.trim()
    const now = Date.now()
    if (trimmed) {
      await saveMessageNote({
        id: `${sessionId}:${message.id}`,
        sessionId,
        messageId: message.id,
        content: trimmed,
        createdAt: now,
        updatedAt: now,
      })
      setNote(trimmed)
      triggerHaptic('success')
      toast.success('Note saved')
    } else if (note) {
      await removeMessageNote(sessionId, message.id)
      setNote(null)
      toast.info('Note removed')
    }
    setNoteEditorOpen(false)
    setNoteDraft('')
  }

  const canNote = !!sessionId && !!message.id
  const noteAction: ContextAction | null = canNote
    ? {
        icon: 'sticky-note' as IconName,
        label: note ? 'Edit note' : 'Add note',
        onPress: handleOpenNoteEditor,
      }
    : null

  const handleToggleReaction = async (emoji: ReactionEmoji) => {
    const updated = await toggleReaction(message.id, emoji)
    setReactions(updated)
    setShowReactionPicker(false)
    triggerHaptic('light')
  }

  const handleSaveToKnowledge = async () => {
    setShowContextMenu(false)
    const ok = await saveToKnowledge(
      message.content,
      message.role as 'user' | 'assistant',
      message.id,
    )
    if (ok) {
      setSavedToKnowledge(true)
      triggerHaptic('success')
      toast.success('Saved to knowledge base')
    } else {
      toast.info('Already in knowledge base')
    }
  }

  const contextActions: ContextAction[] = [
    { icon: 'copy', label: 'Copy', onPress: handleCopy },
    {
      icon: 'external-link',
      label: 'Share',
      onPress: async () => {
        setShowContextMenu(false)
        await Share.share({ message: message.content })
      },
    },
    ...(onReply
      ? [
          {
            icon: 'reply' as IconName,
            label: 'Reply',
            onPress: () => {
              setShowContextMenu(false)
              onReply()
            },
          },
        ]
      : []),
    ...(onForward
      ? [
          {
            icon: 'forward' as IconName,
            label: 'Forward',
            onPress: () => {
              setShowContextMenu(false)
              onForward()
            },
          },
        ]
      : []),
    ...(onBranch
      ? [
          {
            icon: 'git-branch' as IconName,
            label: 'Branch from here',
            onPress: () => {
              setShowContextMenu(false)
              onBranch()
            },
          },
        ]
      : []),
    ...(isUser && onEdit
      ? [
          {
            icon: 'edit' as IconName,
            label: 'Edit',
            onPress: () => {
              setShowContextMenu(false)
              onEdit(message.content)
            },
          },
        ]
      : []),
    {
      icon: pinned ? ('pin' as IconName) : ('pin-off' as IconName),
      label: pinned ? 'Unpin' : 'Pin',
      onPress: handleTogglePin,
    },
    {
      icon: bookmarked ? ('star' as IconName) : ('star-outline' as IconName),
      label: bookmarked ? 'Remove bookmark' : 'Bookmark',
      onPress: handleToggleBookmark,
    },
    {
      icon: savedToKnowledge ? ('check' as IconName) : ('book-open' as IconName),
      label: savedToKnowledge ? 'In knowledge base' : 'Save to knowledge',
      onPress: handleSaveToKnowledge,
    },
    ...(noteAction ? [noteAction] : []),
    {
      icon: 'smile-plus' as IconName,
      label: 'React',
      onPress: () => {
        setShowContextMenu(false)
        setShowReactionPicker(true)
      },
    },
    ...(isUser
      ? []
      : [
          {
            icon: 'thumbs-up' as IconName,
            label: 'Good response',
            onPress: () => {
              setShowContextMenu(false)
              onFeedback?.(true)
            },
          },
          {
            icon: 'thumbs-down' as IconName,
            label: 'Bad response',
            onPress: () => {
              setShowContextMenu(false)
              onFeedback?.(false)
            },
          },
          {
            icon: 'refresh-cw' as IconName,
            label: 'Regenerate',
            onPress: () => {
              setShowContextMenu(false)
              onRegenerate?.()
            },
          },
        ]),
    { icon: 'trash-2' as IconName, label: 'Delete', destructive: true, onPress: handleDelete },
  ]

  const lastTap = useRef<number>(0)

  const handleDoubleTap = () => {
    const now = Date.now()
    if (now - lastTap.current < 300) {
      copyToClipboard(message.content).then((ok) => {
        if (ok) {
          triggerHaptic('success')
          toast.success('Copied to clipboard')
        }
      })
    }
    lastTap.current = now
  }

  return (
    <Animated.View style={[styles.row, { opacity, transform: [{ translateY }] }]}>
      <View style={[styles.innerRow, isUser && styles.innerRowUser]}>
        <Pressable
          style={styles.deleteBtn}
          onPress={handleDelete}
          accessibilityLabel="Delete message"
          accessibilityRole="button"
        >
          <Icon name="trash-2" size={16} color="white" />
        </Pressable>

        {selectMode && (
          <Pressable
            style={styles.selectCheckbox}
            onPress={onSelect}
            accessibilityLabel={selected ? 'Deselect' : 'Select'}
            accessibilityRole="checkbox"
          >
            <View style={[styles.checkbox, selected && styles.checkboxSelected]}>
              {selected && <Icon name="check" size={12} color="white" />}
            </View>
          </Pressable>
        )}

        <Animated.View
          style={[styles.swipeable, { transform: [{ translateX }] }]}
          {...panResponder.panHandlers}
        >
          <Pressable
            style={[
              styles.bubble,
              isUser ? styles.userBubble : styles.assistantBubble,
              highlight && styles.highlight,
              pinned && styles.pinnedBubble,
            ]}
            onPress={selectMode ? onSelect : handleDoubleTap}
            onLongPress={selectMode ? onSelect : onLongPressSelect || handleLongPress}
          >
            {pinned && (
              <View style={styles.pinIcon}>
                <Icon name="pin" size={12} color={primary} />
              </View>
            )}
            {message.images && message.images.length > 0 && (
              <View style={styles.imageContainer}>
                {message.images.map((uri, i) => (
                  <Image key={i} source={{ uri }} style={styles.messageImage} resizeMode="cover" />
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
              <Markdown content={message.content} highlight={searchQuery} />
            ) : (
              <Markdown
                content={
                  collapsed && message.content.length > 500
                    ? message.content.slice(0, 500) + '...'
                    : message.content || 'Thinking...'
                }
                streaming={streaming}
                highlight={searchQuery}
              />
            )}
            {!isUser && !streaming && message.content && message.content.length > 500 && (
              <View style={{ marginTop: 4 }}>
                <Pressable onPress={() => setCollapsed((c) => !c)} hitSlop={8}>
                  <Text style={[styles.timestamp, { color: primary }]}>
                    {collapsed ? 'Show full response' : 'Show less'}
                  </Text>
                </Pressable>
              </View>
            )}
            {noteEditorOpen ? (
              <View style={styles.noteEditor}>
                <TextInput
                  style={styles.noteInput}
                  value={noteDraft}
                  onChangeText={setNoteDraft}
                  placeholder="Add a note..."
                  placeholderTextColor={textMuted}
                  multiline
                  autoFocus
                  accessibilityLabel="Add a note"
                />
                <View style={styles.noteActions}>
                  <Pressable style={styles.noteSaveBtn} onPress={handleSaveNote}>
                    <Text style={styles.noteSaveLabel}>Save</Text>
                  </Pressable>
                  <Pressable
                    onPress={() => {
                      setNoteEditorOpen(false)
                      setNoteDraft('')
                    }}
                  >
                    <Text style={[styles.timestamp, { color: textMuted }]}>Cancel</Text>
                  </Pressable>
                </View>
              </View>
            ) : note ? (
              <Pressable style={styles.noteStrip} onPress={handleOpenNoteEditor}>
                <Icon name="sticky-note" size={12} color={primary} />
                <Text style={styles.noteText}>{note}</Text>
              </Pressable>
            ) : null}
          </Pressable>
        </Animated.View>
      </View>

      <Pressable
        onPress={() => setShowFullDate((d) => !d)}
        style={[styles.timestampRow, isUser && styles.timestampRowUser]}
      >
        <Text style={[styles.timestamp, isUser && styles.timestampUser]}>
          {showFullDate
            ? new Date(message.timestamp).toLocaleString()
            : formatTime(message.timestamp)}
          {message.content && message.content.length > 200 && (
            <Text style={styles.readTime}>
              {' '}
              · {Math.max(1, Math.ceil(message.content.split(/\s+/).length / 200))} min read
            </Text>
          )}
        </Text>
      </Pressable>

      {/* Compact action bar for AI messages */}
      {!isUser && !streaming && message.content && (
        <View
          style={{
            flexDirection: 'row',
            alignItems: 'center',
            gap: 2,
            marginTop: 6,
            marginLeft: 2,
          }}
        >
          <Pressable
            style={{ paddingHorizontal: 8, paddingVertical: 5, borderRadius: 6 }}
            accessibilityLabel="Copy message"
            onPress={() => {
              copyToClipboard(message.content || '')
              triggerHaptic('light')
              toast.success('Copied')
            }}
          >
            <Icon name="copy" size={14} color={textMuted} />
          </Pressable>
          {onFeedback && (
            <>
              <Pressable
                style={{ paddingHorizontal: 8, paddingVertical: 5, borderRadius: 6 }}
                accessibilityLabel="Good response"
                onPress={() => {
                  triggerHaptic('light')
                  onFeedback(true)
                }}
              >
                <Icon name="thumbs-up" size={14} color={textMuted} />
              </Pressable>
              <Pressable
                style={{ paddingHorizontal: 8, paddingVertical: 5, borderRadius: 6 }}
                accessibilityLabel="Bad response"
                onPress={() => {
                  triggerHaptic('light')
                  onFeedback(false)
                }}
              >
                <Icon name="thumbs-down" size={14} color={textMuted} />
              </Pressable>
            </>
          )}
          {onRegenerate && (
            <Pressable
              style={{ paddingHorizontal: 8, paddingVertical: 5, borderRadius: 6 }}
              accessibilityLabel="Regenerate response"
              onPress={() => {
                triggerHaptic('medium')
                onRegenerate()
              }}
            >
              <Icon name="refresh-cw" size={14} color={textMuted} />
            </Pressable>
          )}
        </View>
      )}

      {message.status === 'failed' && (
        <View style={{ flexDirection: 'row', alignItems: 'center', gap: 6, marginTop: 4 }}>
          <Icon name="triangle-alert" size={14} color={colors.error} />
          <Text style={[styles.timestamp, { color: colors.error }]}>Failed to send</Text>
          {onRetry && (
            <Pressable onPress={onRetry} hitSlop={8}>
              <View style={{ flexDirection: 'row', alignItems: 'center', gap: 3 }}>
                <Icon name="refresh-cw" size={12} color={primary} />
                <Text style={[styles.timestamp, { color: primary }]}>Retry</Text>
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
              onLongPress={() => setShowReactionPicker(true)}
            >
              <Text style={styles.reactionEmoji}>{emoji}</Text>
            </Pressable>
          ))}
        </View>
      )}

      {showReactionPicker && (
        <View style={[styles.reactionPicker, isUser && styles.reactionPickerUser]}>
          {REACTION_EMOJIS.map((emoji) => (
            <Pressable
              key={emoji}
              style={[
                styles.reactionOption,
                reactions.includes(emoji) && styles.reactionOptionActive,
              ]}
              onPress={() => handleToggleReaction(emoji)}
            >
              <Text style={styles.reactionOptionText}>{emoji}</Text>
            </Pressable>
          ))}
          <Pressable
            style={styles.reactionClose}
            onPress={() => setShowReactionPicker(false)}
            accessibilityLabel="Close reaction picker"
            accessibilityRole="button"
          >
            <Icon name="x" size={14} color={textMuted} />
          </Pressable>
        </View>
      )}

      <Modal
        visible={showContextMenu}
        transparent
        animationType="fade"
        onRequestClose={() => setShowContextMenu(false)}
      >
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
                style={[
                  styles.contextAction,
                  action.destructive && styles.contextActionDestructive,
                ]}
                onPress={action.onPress}
              >
                <Icon name={action.icon} size={18} color={textColor} />
                <Text
                  style={[
                    styles.contextLabel,
                    action.destructive && styles.contextLabelDestructive,
                  ]}
                >
                  {action.label}
                </Text>
              </Pressable>
            ))}
          </View>
        </Pressable>
      </Modal>
    </Animated.View>
  )
}
