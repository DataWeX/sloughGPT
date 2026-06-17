import React, {useEffect, useRef, useState, useCallback} from 'react';
import {
  View,
  FlatList,
  Text,
  StyleSheet,
  KeyboardAvoidingView,
  Platform,
  TouchableOpacity,
  ActivityIndicator,
} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {useChatStore} from '../../stores/chat-store';
import {useModelStore} from '../../stores/model-store';
import {MessageBubble} from '../../components/MessageBubble';
import {ChatInput} from '../../components/ChatInput';
import {colors, spacing, radii, typography} from '../../theme';
import type {Message} from '../../types';

const SUGGESTIONS = [
  'Tell me something interesting',
  'Help me brainstorm',
  'Explain a concept',
];

export function ChatScreen() {
  const {
    messages,
    streaming,
    error,
    sendMessage,
    regenerate,
    cancelStream,
    recordFeedback,
    clearError,
    refreshSessions,
  } = useChatStore();
  const {health, currentSoul} = useModelStore();
  const flatListRef = useRef<FlatList>(null);
  const [atBottom, setAtBottom] = useState(true);

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

  const isConnected = health?.status === 'healthy';

  return (
    <SafeAreaView style={styles.safe} edges={['top']}>
      <KeyboardAvoidingView
        style={styles.flex}
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
        keyboardVerticalOffset={0}>
        <View style={styles.header}>
          <View style={styles.headerLeft}>
            <Text style={styles.title}>Chat</Text>
            {currentSoul && (
              <View style={styles.soulPill}>
                <Text style={styles.soulText}>{currentSoul.name}</Text>
              </View>
            )}
          </View>
          <View style={styles.headerRight}>
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
});
