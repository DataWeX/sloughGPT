import React, {useEffect, useState, useCallback, useRef} from 'react';
import {
  View,
  FlatList,
  Text,
  StyleSheet,
  KeyboardAvoidingView,
  Platform,
  TouchableOpacity,
  ActivityIndicator,
  Pressable,
  Modal,
  TextInput,
  Alert,
  Keyboard,
  Share,
} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {useChatStore} from '../stores/chat-store';
import {useModelStore} from '../stores/model-store';
import {useOnlineStatus} from '../hooks/useOnlineStatus';
import {MessageBubble} from '../components/MessageBubble';
import {ChatInput} from '../components/ChatInput';
import {StatusBadge} from '../components/StatusBadge';
import {colors, spacing, radii, typography} from '../theme';
import type {Message, Session} from '../types';

const SUGGESTIONS = [
  'Tell me something interesting',
  'Help me brainstorm',
  'Explain a concept',
];

export function ChatScreen() {
  const {
    sessions,
    messages,
    streaming,
    error,
    sendMessage,
    regenerate,
    cancelStream,
    recordFeedback,
    clearError,
    refreshSessions,
    loadSession,
    deleteSession,
    createSession,
  } = useChatStore();
  const {health, currentSoul, souls, switchSoul} = useModelStore();
  const online = useOnlineStatus();
  const flatListRef = useRef<FlatList>(null);
  const [atBottom, setAtBottom] = useState(true);
  const [showDrawer, setShowDrawer] = useState(false);
  const [showSoulPicker, setShowSoulPicker] = useState(false);
  const [showSettings, setShowSettings] = useState(false);

  useEffect(() => {
    refreshSessions();
  }, []);

  useEffect(() => {
    if (atBottom && messages.length > 0) {
      setTimeout(() => {
        flatListRef.current?.scrollToEnd({animated: true});
      }, 50);
    }
  }, [messages, atBottom]);

  const handleSend = useCallback(
    (text: string) => {
      sendMessage(text);
    },
    [sendMessage],
  );

  const handleSuggestion = useCallback(
    (s: string) => {
      sendMessage(s);
    },
    [sendMessage],
  );

  const renderItem = useCallback(
    ({item}: {item: Message}) => (
      <MessageBubble
        message={item}
        onRegenerate={
          item.role === 'assistant' ? () => regenerate(item.id) : undefined
        }
        onFeedback={
          item.role === 'assistant'
            ? positive => recordFeedback(item.id, positive)
            : undefined
        }
      />
    ),
    [regenerate, recordFeedback],
  );

  const keyExtractor = useCallback((item: Message) => item.id, []);

  const onScroll = useCallback((e: any) => {
    const {contentOffset, contentSize, layoutMeasurement} = e.nativeEvent;
    const distFromBottom =
      contentSize.height - layoutMeasurement.height - contentOffset.y;
    setAtBottom(distFromBottom < 50);
  }, []);

  const isConnected = online;

  return (
    <SafeAreaView style={styles.safe} edges={['top']}>
      <KeyboardAvoidingView
        style={styles.flex}
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
        keyboardVerticalOffset={0}>
        <View style={styles.header}>
          <View style={styles.headerLeft}>
            <TouchableOpacity onPress={() => setShowDrawer(true)}>
              <Text style={styles.menuBtn}>☰</Text>
            </TouchableOpacity>
            <TouchableOpacity onPress={() => setShowSoulPicker(true)}>
              <Text style={styles.menuBtn}>⚙</Text>
            </TouchableOpacity>
            <Text style={styles.title}>Chat</Text>
            {currentSoul && (
              <TouchableOpacity
                style={styles.soulPill}
                onPress={() => setShowSoulPicker(true)}>
                <Text style={styles.soulText}>{currentSoul.name}</Text>
              </TouchableOpacity>
            )}
          </View>
          <View style={styles.headerRight}>
            {messages.length > 0 && (
              <TouchableOpacity
                style={styles.exportBtn}
                onPress={async () => {
                  const md = messages.map(m => {
                    const role = m.role === 'user' ? 'You' : 'Assistant';
                    return `**${role}:**\n${m.content}`;
                  }).join('\n\n---\n\n');
                  await Share.share({title: 'Chat Export', message: md});
                }}>
                <Text style={styles.exportBtnText}>Export</Text>
              </TouchableOpacity>
            )}
            <TouchableOpacity
              style={styles.newChatBtn}
              onPress={() => createSession()}>
              <Text style={styles.newChatText}>+ New</Text>
            </TouchableOpacity>
            <View
              style={[
                styles.dot,
                {backgroundColor: isConnected ? colors.success : colors.error},
              ]}
            />
          </View>
        </View>

        {error && (
          <TouchableOpacity style={styles.errorBanner} onPress={clearError}>
            <Text style={styles.errorText}>{error}</Text>
            <Text style={styles.errorDismiss}>×</Text>
          </TouchableOpacity>
        )}

        {messages.length === 0 && !streaming ? (
          <View style={styles.emptyContainer}>
            <Text style={styles.emptyEmoji}>💬</Text>
            <Text style={styles.emptyTitle}>Start a conversation</Text>
            <Text style={styles.emptySubtitle}>
              {currentSoul
                ? `Chatting with ${currentSoul.name}`
                : 'Type a message below to begin'}
            </Text>
            <View style={styles.suggestions}>
              {SUGGESTIONS.map(s => (
                <TouchableOpacity
                  key={s}
                  style={styles.suggestionChip}
                  onPress={() => handleSuggestion(s)}>
                  <Text style={styles.suggestionText}>{s}</Text>
                </TouchableOpacity>
              ))}
            </View>
          </View>
        ) : (
          <Pressable style={{flex: 1}} onPress={() => Keyboard.dismiss()}>
            <FlatList
              ref={flatListRef}
              data={messages}
              renderItem={renderItem}
              keyExtractor={keyExtractor}
              contentContainerStyle={styles.messageList}
              onScroll={onScroll}
              scrollEventThrottle={16}
              onContentSizeChange={() => {
              if (atBottom) {
                flatListRef.current?.scrollToEnd({animated: false});
              }
            }}
            />
          </Pressable>
        )}

        {streaming && (
          <View style={styles.thinkingRow}>
            <ActivityIndicator size="small" color={colors.primary} />
            <Text style={styles.thinkingText}>Thinking...</Text>
          </View>
        )}

        <ChatInput
          onSend={handleSend}
          disabled={streaming}
          onStop={cancelStream}
        />

        {!atBottom && messages.length > 0 && (
          <TouchableOpacity
            style={styles.jumpBtn}
            onPress={() =>
              flatListRef.current?.scrollToEnd({animated: true})
            }>
            <Text style={styles.jumpText}>↓</Text>
          </TouchableOpacity>
        )}
      </KeyboardAvoidingView>

      {/* Session drawer */}
      <Modal visible={showDrawer} animationType="slide" transparent>
        <View style={styles.drawerOverlay}>
          <View style={styles.drawer}>
            <View style={styles.drawerHeader}>
              <Text style={styles.drawerTitle}>Conversations</Text>
              <TouchableOpacity onPress={() => setShowDrawer(false)}>
                <Text style={styles.drawerClose}>×</Text>
              </TouchableOpacity>
            </View>
            <FlatList
              data={sessions}
              keyExtractor={item => item.id}
              renderItem={({item: session}) => (
                <TouchableOpacity
                  style={[
                    styles.sessionItem,
                    session.id === useChatStore.getState().activeSessionId &&
                      styles.sessionItemActive,
                  ]}
                  onPress={() => {
                    loadSession(session.id);
                    setShowDrawer(false);
                  }}>
                  <View style={styles.sessionInfo}>
                    <Text style={styles.sessionTitle} numberOfLines={1}>
                      {session.title || 'New conversation'}
                    </Text>
                    <Text style={styles.sessionMeta}>
                      {session.message_count || 0} messages
                    </Text>
                  </View>
                  <TouchableOpacity
                    onPress={() => {
                      Alert.alert('Delete', 'Delete this conversation?', [
                        {text: 'Cancel', style: 'cancel'},
                        {
                          text: 'Delete',
                          style: 'destructive',
                          onPress: () => deleteSession(session.id),
                        },
                      ]);
                    }}>
                    <Text style={styles.sessionDelete}>×</Text>
                  </TouchableOpacity>
                </TouchableOpacity>
              )}
              ListEmptyComponent={
                <Text style={styles.drawerEmpty}>
                  No conversations yet
                </Text>
              }
            />
          </View>
        </View>
      </Modal>

      {/* Soul picker */}
      <Modal visible={showSoulPicker} animationType="slide" transparent>
        <TouchableOpacity
          style={styles.drawerOverlay}
          activeOpacity={1}
          onPress={() => setShowSoulPicker(false)}>
          <TouchableOpacity activeOpacity={1} style={styles.drawer} onPress={() => {}}>
            <View style={styles.drawerHeader}>
              <Text style={styles.drawerTitle}>Personality</Text>
              <TouchableOpacity onPress={() => setShowSoulPicker(false)}>
                <Text style={styles.drawerClose}>×</Text>
              </TouchableOpacity>
            </View>
            <FlatList
              data={souls}
              keyExtractor={item => item.name}
              renderItem={({item: soul}) => (
                <TouchableOpacity
                  style={[
                    styles.sessionItem,
                    soul.name === currentSoul?.name && styles.sessionItemActive,
                  ]}
                  onPress={() => {
                    switchSoul(soul.name);
                    setShowSoulPicker(false);
                  }}>
                  <View style={styles.sessionInfo}>
                    <Text style={styles.sessionTitle}>{soul.name}</Text>
                    {soul.description && (
                      <Text style={styles.sessionMeta} numberOfLines={2}>
                        {soul.description}
                      </Text>
                    )}
                  </View>
                  {soul.name === currentSoul?.name && (
                    <Text style={styles.checkMark}>✓</Text>
                  )}
                </TouchableOpacity>
              )}
              ListEmptyComponent={
                <Text style={styles.drawerEmpty}>No personalities available</Text>
              }
            />
          </TouchableOpacity>
        </TouchableOpacity>
      </Modal>

      {/* Soul picker modal */}
      <Modal visible={showSoulPicker} animationType="slide" transparent>
        <View style={styles.drawerOverlay}>
          <View style={styles.drawer}>
            <View style={styles.drawerHeader}>
              <Text style={styles.drawerTitle}>Personality</Text>
              <TouchableOpacity onPress={() => setShowSoulPicker(false)}>
                <Text style={styles.drawerClose}>×</Text>
              </TouchableOpacity>
            </View>
            {currentSoul && (
              <View style={styles.activeSoul}>
                <Text style={styles.activeSoulLabel}>Active</Text>
                <Text style={styles.activeSoulName}>{currentSoul.name}</Text>
                {currentSoul.description && (
                  <Text style={styles.activeSoulDesc}>{currentSoul.description}</Text>
                )}
              </View>
            )}
            <FlatList
              data={souls}
              keyExtractor={item => item.name}
              renderItem={({item: soul}) => {
                const isActive = currentSoul?.name === soul.name;
                return (
                  <TouchableOpacity
                    style={[styles.sessionItem, isActive && styles.sessionItemActive]}
                    onPress={() => {
                      switchSoul(soul.name);
                      setShowSoulPicker(false);
                    }}>
                    <View style={styles.sessionInfo}>
                      <Text style={styles.sessionTitle}>{soul.name}</Text>
                      {soul.description && (
                        <Text style={styles.sessionMeta} numberOfLines={1}>
                          {soul.description}
                        </Text>
                      )}
                      {soul.traits && soul.traits.length > 0 && (
                        <View style={styles.traitRow}>
                          {soul.traits.map(trait => (
                            <StatusBadge key={trait} label={trait} variant="info" />
                          ))}
                        </View>
                      )}
                    </View>
                    {isActive && <Text style={styles.checkMark}>✓</Text>}
                  </TouchableOpacity>
                );
              }}
              ListEmptyComponent={
                <Text style={styles.drawerEmpty}>No personalities found</Text>
              }
            />
          </View>
        </View>
      </Modal>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: colors.background,
  },
  flex: {
    flex: 1,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  headerLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  menuBtn: {
    fontSize: 20,
    color: colors.textSecondary,
    padding: spacing.xs,
  },
  title: {
    ...typography.h3,
    color: colors.text,
  },
  soulPill: {
    backgroundColor: colors.primaryLight + '30',
    paddingHorizontal: spacing.sm,
    paddingVertical: 2,
    borderRadius: radii.full,
  },
  soulText: {
    ...typography.small,
    color: colors.primary,
  },
  headerRight: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  exportBtn: {
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
    borderRadius: radii.md,
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.border,
  },
  exportBtnText: {
    ...typography.small,
    color: colors.textSecondary,
    fontWeight: '500',
  },
  newChatBtn: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.xs,
    borderRadius: radii.md,
    backgroundColor: colors.primary + '15',
  },
  newChatText: {
    ...typography.small,
    color: colors.primary,
    fontWeight: '600',
  },
  dot: {
    width: 8,
    height: 8,
    borderRadius: radii.full,
  },
  errorBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: '#FDE8E8',
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
  },
  errorText: {
    ...typography.caption,
    color: colors.error,
    flex: 1,
  },
  errorDismiss: {
    ...typography.body,
    color: colors.error,
    fontWeight: '600',
    marginLeft: spacing.sm,
  },
  messageList: {
    paddingTop: spacing.md,
    paddingBottom: spacing.sm,
  },
  thinkingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.xs,
    gap: spacing.xs,
  },
  thinkingText: {
    ...typography.caption,
    color: colors.textMuted,
  },
  emptyContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: spacing.xxxl,
  },
  emptyEmoji: {
    fontSize: 48,
    marginBottom: spacing.lg,
  },
  emptyTitle: {
    ...typography.h2,
    color: colors.text,
    marginBottom: spacing.sm,
  },
  emptySubtitle: {
    ...typography.body,
    color: colors.textSecondary,
    textAlign: 'center',
    marginBottom: spacing.xxl,
  },
  suggestions: {
    gap: spacing.sm,
    alignItems: 'center',
  },
  suggestionChip: {
    backgroundColor: colors.surface,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
    borderRadius: radii.full,
    borderWidth: 1,
    borderColor: colors.border,
  },
  suggestionText: {
    ...typography.caption,
    color: colors.textSecondary,
  },
  jumpBtn: {
    position: 'absolute',
    right: spacing.lg,
    bottom: 80,
    width: 36,
    height: 36,
    borderRadius: radii.full,
    backgroundColor: colors.primary,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 2},
    shadowOpacity: 0.15,
    shadowRadius: 4,
    elevation: 3,
  },
  jumpText: {
    color: colors.white,
    fontSize: 18,
    fontWeight: '600',
  },
  // Drawer
  drawerOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.4)',
    justifyContent: 'flex-start',
  },
  drawer: {
    width: '80%',
    maxWidth: 320,
    maxHeight: '80%',
    backgroundColor: colors.background,
    borderBottomRightRadius: radii.lg,
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 4},
    shadowOpacity: 0.2,
    shadowRadius: 8,
    elevation: 5,
  },
  drawerHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  drawerTitle: {
    ...typography.h3,
    color: colors.text,
  },
  drawerClose: {
    fontSize: 24,
    color: colors.textMuted,
    padding: spacing.xs,
  },
  sessionItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  sessionItemActive: {
    backgroundColor: colors.primary + '10',
  },
  sessionInfo: {flex: 1},
  sessionTitle: {
    ...typography.body,
    color: colors.text,
    fontWeight: '500',
  },
  sessionMeta: {
    ...typography.small,
    color: colors.textMuted,
  },
  sessionDelete: {
    fontSize: 18,
    color: colors.textMuted,
    padding: spacing.xs,
  },
  checkMark: {
    fontSize: 16,
    color: colors.primary,
    fontWeight: '700',
    padding: spacing.xs,
  },
  drawerEmpty: {
    ...typography.caption,
    color: colors.textMuted,
    textAlign: 'center',
    padding: spacing.xxxl,
  },
});
