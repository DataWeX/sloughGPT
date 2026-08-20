import React, {useCallback, useRef, useEffect} from 'react';
import {
  FlatList,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  TextInput as RNTextInput,
  Keyboard,
  RefreshControl,
} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {YStack, XStack, Text} from 'tamagui';
import {useColors} from '../theme/colors';
import {useChatActions} from '../hooks/useChatActions';
import {ChatDrawer} from '../components/ChatDrawer';
import {ChatBottomSheets} from '../components/ChatBottomSheets';
import {useSidebar} from '../contexts/SidebarContext';
import {MessageBubble} from '../components/MessageBubble';
import {ChatInput} from '../components/ChatInput';
import {ReasoningPanel} from '../components/ReasoningPanel';
import {Icon} from '../components/Icon';

const SUGGESTIONS = [
  {icon: 'zap' as const, text: 'Tell me something interesting', prompt: 'Tell me something interesting'},
  {icon: 'target' as const, text: 'Help me brainstorm ideas', prompt: 'Help me brainstorm ideas for a project'},
  {icon: 'book-open' as const, text: 'Explain a concept', prompt: 'Explain a concept to me in simple terms'},
  {icon: 'terminal' as const, text: 'Write some code', prompt: 'Write some code for me'},
];

export function ChatScreen() {
  const colors = useColors();
  const flatListRef = useRef<FlatList>(null);
  const a = useChatActions(flatListRef);
  const {open: openSidebar} = useSidebar();

  useEffect(() => {
    const {Keyboard: KB} = require('react-native');
    const sub = KB.addListener('keyboardDidHide', () => {
      // Only dismiss keyboard-triggered modals (search), not explicit ones (drawer, soul picker)
      a.dismissKeyboardModals();
    });
    return () => sub.remove();
  }, [a.dismissKeyboardModals]);

  return (
    <SafeAreaView style={{flex: 1, backgroundColor: 'var(--background)'}} edges={['top']}>
      <KeyboardAvoidingView
        style={{flex: 1}}
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
        keyboardVerticalOffset={0}>
        {/* Header */}
        <XStack
          paddingHorizontal={16}
          paddingVertical={12}
          borderBottomWidth={0.5}
          borderBottomColor="$borderColor"
          backgroundColor="$background"
          alignItems="center"
          justifyContent="space-between">
          <XStack alignItems="center" gap={12}>
            <YStack
              width={36} height={36} borderRadius={12}
              alignItems="center" justifyContent="center"
              onPress={openSidebar}
              pressStyle={{opacity: 0.6, scale: 0.95}}
              accessible accessibilityRole="button" accessibilityLabel="Open menu">
              <Icon name="menu" size={18} color={colors.textSecondary} />
            </YStack>
            <YStack
              alignItems="flex-start"
              onPress={() => {
                const now = Date.now();
                if (now - a.lastHeaderTap.current < 300) {
                  flatListRef.current?.scrollToOffset({offset: 0, animated: true});
                }
                a.lastHeaderTap.current = now;
              }}>
              <XStack alignItems="center" gap={6}>
                <Text fontSize={17} fontWeight="700" letterSpacing={-0.3} color="$color">Chat</Text>
                {a.currentSoul && (
                  <YStack
                    backgroundColor={colors.primaryAlpha(0.1)}
                    paddingHorizontal={10}
                    paddingVertical={3}
                    borderRadius={999}
                    borderWidth={0.5}
                    borderColor={colors.primaryAlpha(0.18)}
                    onPress={() => a.setShowSoulPicker(true)}
                    pressStyle={{opacity: 0.7, scale: 0.95}}>
                    <Text fontSize={11} fontWeight="600" color="$color9">{a.currentSoul.name}</Text>
                  </YStack>
                )}
              </XStack>
            </YStack>
          </XStack>

          <XStack alignItems="center" gap={4}>
            {a.messages.length > 0 && (
              <YStack
                width={36} height={36} borderRadius={12}
                alignItems="center" justifyContent="center"
                onPress={a.handleExportChat}
                pressStyle={{opacity: 0.6, scale: 0.95}}
                accessible accessibilityRole="button" accessibilityLabel="Export conversation">
                <Icon name="share-2" size={18} color={colors.textSecondary} />
              </YStack>
            )}
            <YStack
              width={36} height={36} borderRadius={12}
              alignItems="center" justifyContent="center"
              onPress={() => a.setShowSettings(true)}
              pressStyle={{opacity: 0.6, scale: 0.95}}
              accessible accessibilityRole="button" accessibilityLabel="Settings">
              <Icon name="more-vertical" size={18} color={colors.textSecondary} />
            </YStack>
          </XStack>
        </XStack>

        {a.error && (
          <XStack
            paddingHorizontal={16} paddingVertical={10}
            backgroundColor={colors.errorAlpha(0.06)}
            borderBottomWidth={0.5} borderBottomColor={colors.errorAlpha(0.12)}
            alignItems="center" gap={8}
            onPress={a.clearError}>
            <YStack width={6} height={6} borderRadius={3} backgroundColor="#EF4444" />
            <Text fontSize={12} color="#EF4444" flex={1} numberOfLines={2}>{a.error}</Text>
            <Icon name="x" size={14} color="#EF4444" />
          </XStack>
        )}

        {!a.online && (
          <XStack
            paddingHorizontal={16} paddingVertical={8}
            borderBottomWidth={0.5} borderBottomColor="$borderColor"
            alignItems="center" gap={6}>
            <YStack width={5} height={5} borderRadius={3} backgroundColor="#F59E0B" />
            <Text fontSize={11} fontWeight="500" color="$color10">Offline</Text>
            {a.offlineQueue > 0 && (
              <YStack onPress={a.retryPendingSends}>
                <Text fontSize={11} color="$color11" textDecorationLine="underline">{a.offlineQueue} queued</Text>
              </YStack>
            )}
          </XStack>
        )}

        {a.showSearch && (
          <XStack
            paddingHorizontal={16} paddingVertical={8}
            gap={8} alignItems="center">
            <RNTextInput
              style={{
                flex: 1, fontSize: 13,
                color: colors.text,
                backgroundColor: colors.primaryAlpha(0.06),
                borderRadius: 10,
                paddingHorizontal: 12, paddingVertical: 7,
                borderWidth: 0.5, borderColor: colors.border,
              }}
              value={a.searchQuery}
              onChangeText={a.setSearchQuery}
              placeholder="Search messages..."
              placeholderTextColor={colors.textMuted}
              autoFocus
            />
            {a.matchCount > 0 && (
              <Text fontSize={11} color={colors.textMuted} minWidth={40} textAlign="center">
                {a.currentMatchIdx + 1}/{a.matchCount}
              </Text>
            )}
            {a.matchCount > 0 && (
              <>
                <YStack
                  width={28} height={28} borderRadius={9}
                  alignItems="center" justifyContent="center"
                  backgroundColor={colors.primaryAlpha(0.06)}
                  onPress={a.searchPrev}
                  pressStyle={{opacity: 0.6}}
                  accessible accessibilityRole="button" accessibilityLabel="Previous match">
                  <Icon name="chevron-up" size={14} color={colors.textMuted} />
                </YStack>
                <YStack
                  width={28} height={28} borderRadius={9}
                  alignItems="center" justifyContent="center"
                  backgroundColor={colors.primaryAlpha(0.06)}
                  onPress={a.searchNext}
                  pressStyle={{opacity: 0.6}}
                  accessible accessibilityRole="button" accessibilityLabel="Next match">
                  <Icon name="chevron-down" size={14} color={colors.textMuted} />
                </YStack>
              </>
            )}
            <YStack
              width={28} height={28} borderRadius={9}
              alignItems="center" justifyContent="center"
              backgroundColor={colors.primaryAlpha(0.06)}
              onPress={() => { a.setShowSearch(false); a.setSearchQuery(''); }}
              pressStyle={{opacity: 0.6}}
              accessible accessibilityRole="button" accessibilityLabel="Close search">
              <Icon name="x" size={14} color={colors.textMuted} />
            </YStack>
          </XStack>
        )}

        {a.messages.length === 0 && !a.streaming ? (
          <YStack
            flex={1} alignItems="center" justifyContent="center"
            paddingHorizontal={32}
            backgroundColor={a.chatBackground || 'var(--background)'}>
            <YStack
              width={80} height={80} borderRadius={40}
              backgroundColor={colors.primaryAlpha(0.08)}
              alignItems="center" justifyContent="center"
              marginBottom={24}
              borderWidth={0.5}
              borderColor={colors.primaryAlpha(0.15)}>
              <YStack
                width={56} height={56} borderRadius={28}
                backgroundColor={colors.primaryAlpha(0.1)}
                alignItems="center" justifyContent="center">
                <Icon name="message-circle" size={26} color={colors.primary} />
              </YStack>
            </YStack>

            <Text fontSize={22} fontWeight="700" letterSpacing={-0.5} color="$color" marginBottom={8} textAlign="center">
              {a.currentSoul ? a.currentSoul.name : 'Chat'}
            </Text>

            <Text fontSize={14} color="$color11" textAlign="center" marginBottom={36} lineHeight={20} maxWidth={260}>
              {a.currentSoul
                ? (a.currentSoul.description || 'Ask me anything')
                : 'Start a conversation'}
            </Text>

            <YStack gap={12} width="100%" maxWidth={320}>
              {SUGGESTIONS.map(s => (
                <YStack
                  key={s.text}
                  flexDirection="row"
                  alignItems="center"
                  gap={12}
                  paddingHorizontal={16}
                  paddingVertical={14}
                  borderRadius={14}
                  backgroundColor={colors.primaryAlpha(0.06)}
                  borderWidth={0.5}
                  borderColor={colors.primaryAlpha(0.12)}
                  onPress={() => a.handleSuggestion(s.prompt)}
                  pressStyle={{opacity: 0.7, scale: 0.98, backgroundColor: colors.primaryAlpha(0.1)}}>
                  <YStack
                    width={32} height={32} borderRadius={10}
                    backgroundColor={colors.primaryAlpha(0.1)}
                    alignItems="center" justifyContent="center">
                    <Icon name={s.icon} size={16} color={colors.primary} />
                  </YStack>
                  <Text fontSize={14} fontWeight="500" color="$color" flex={1}>{s.text}</Text>
                  <YStack style={{transform: [{rotate: '-90deg'}]}}>
                    <Icon name="chevron-down" size={14} color={colors.textMuted} />
                  </YStack>
                </YStack>
              ))}
            </YStack>

            <Text fontSize={11} color="$color10" marginTop={32} textAlign="center" letterSpacing={0.3}>
              Swipe left on a message to delete it
            </Text>
          </YStack>
        ) : (
          <Pressable
            style={{flex: 1, backgroundColor: a.chatBackground || 'var(--background)'}}
            onPress={() => Keyboard.dismiss()}>
            <FlatList
              ref={flatListRef}
              data={a.messages}
              renderItem={a.renderItem}
              keyExtractor={a.keyExtractor}
              contentContainerStyle={{paddingTop: 8, paddingBottom: 8}}
              onScroll={a.onScroll}
              scrollEventThrottle={16}
              removeClippedSubviews
              maxToRenderPerBatch={10}
              windowSize={11}
              refreshControl={
                <RefreshControl
                  refreshing={a.refreshing}
                  onRefresh={a.onPullRefresh}
                  tintColor={colors.primary}
                />
              }
              onContentSizeChange={() => {
              if (a.atBottom) {
                flatListRef.current?.scrollToEnd({animated: false});
              }
            }}
            />
          </Pressable>
        )}

        {a.pinnedIds.length > 0 && !a.selectMode && (
          <XStack
            paddingHorizontal={16} paddingVertical={8}
            gap={8} alignItems="center"
            onPress={() => {
              const firstPinnedIndex = a.messages.findIndex(m => a.pinnedIds.includes(m.id));
              if (firstPinnedIndex >= 0) {
                flatListRef.current?.scrollToIndex({index: firstPinnedIndex, animated: true, viewPosition: 0});
              }
            }}
            pressStyle={{opacity: 0.7}}>
            <YStack width={24} height={24} borderRadius={8} backgroundColor={colors.primaryAlpha(0.08)} alignItems="center" justifyContent="center">
              <Icon name="pin" size={12} color="$color9" />
            </YStack>
            <Text fontSize={11} fontWeight="500" color="$color9">
              {a.pinnedIds.length} pinned {a.pinnedIds.length === 1 ? 'message' : 'messages'}
            </Text>
          </XStack>
        )}

        {a.selectMode && (
          <XStack
            paddingHorizontal={16} paddingVertical={10}
            borderTopWidth={0.5} borderTopColor="$borderColor"
            alignItems="center" justifyContent="space-between"
            backgroundColor={colors.primaryAlpha(0.04)}>
            <YStack onPress={() => {
              if (a.selectedIds.size === a.messages.length) {
                a.setSelectedIds(new Set());
              } else {
                a.setSelectedIds(new Set(a.messages.map(m => m.id)));
              }
            }} pressStyle={{opacity: 0.6}}>
              <Text fontSize={13} fontWeight="600" color="$color9">
                {a.selectedIds.size === a.messages.length ? 'Deselect all' : 'Select all'}
              </Text>
            </YStack>
            <XStack gap={12} alignItems="center">
              <Text fontSize={12} color="$color10">{a.selectedIds.size} selected</Text>
              <YStack paddingHorizontal={10} paddingVertical={5} borderRadius={8}
                backgroundColor={a.selectedIds.size > 0 ? colors.errorAlpha(0.1) : 'transparent'}
                opacity={a.selectedIds.size === 0 ? 0.4 : 1}
                onPress={a.deleteSelected}
                disabled={a.selectedIds.size === 0}>
                <Text fontSize={13} fontWeight="600"
                  color={a.selectedIds.size > 0 ? colors.error : colors.textMuted}>Delete</Text>
              </YStack>
              <YStack paddingHorizontal={10} paddingVertical={5} borderRadius={8}
                backgroundColor={colors.primaryAlpha(0.08)} onPress={a.toggleSelectMode}>
                <Text fontSize={13} fontWeight="600" color="$color9">Done</Text>
              </YStack>
            </XStack>
          </XStack>
        )}

        <ReasoningPanel visible={a.streaming && a.messages.length > 0 && !a.messages[a.messages.length - 1]?.content} />

        {a.replyTo && (
          <XStack
            paddingHorizontal={16} paddingVertical={8}
            gap={10} alignItems="center"
            backgroundColor={colors.primaryAlpha(0.04)}
            borderTopWidth={0.5} borderTopColor="$borderColor">
            <YStack width={3} height={32} borderRadius={2} backgroundColor="$color9" />
            <YStack flex={1}>
              <Text fontSize={11} fontWeight="600" color="$color9" marginBottom={1}>
                Replying to {a.replyTo.role === 'user' ? 'yourself' : 'assistant'}
              </Text>
              <Text fontSize={12} color="$color10" numberOfLines={1}>
                {a.replyTo.content}
              </Text>
            </YStack>
            <YStack
              width={24} height={24} borderRadius={8}
              backgroundColor={colors.primaryAlpha(0.06)}
              alignItems="center" justifyContent="center"
              onPress={() => a.setReplyTo(null)}
              pressStyle={{opacity: 0.6}}>
              <Icon name="x" size={12} color="$color10" />
            </YStack>
          </XStack>
        )}

        <ChatInput
          onSend={a.handleSend}
          onSendWithImages={a.handleSendWithImages}
          onImage={a.handleImage}
          onVoice={a.handleVoice}
          onFile={a.handleFile}
          disabled={a.streaming}
          onStop={a.cancelStream}
          isRecording={a.isRecording}
          sessionId={a.activeSessionId}
          editText={a.editingMessage}
          onCancelEdit={() => a.setEditingMessage(null)}
          voiceMessageMode={a.voiceMessageMode}
          onVoiceMessageToggle={() => a.setVoiceMessageMode(v => !v)}
        />

        {!a.atBottom && a.messages.length > 0 && (
          <YStack
            position="absolute"
            right={20}
            bottom={80}
            width={40}
            height={40}
            borderRadius={20}
            backgroundColor={colors.primaryAlpha(0.15)}
            borderWidth={0.5}
            borderColor={colors.primaryAlpha(0.25)}
            alignItems="center"
            justifyContent="center"
            shadowColor="$color9"
            shadowOffset={{width: 0, height: 4}}
            shadowOpacity={0.25}
            shadowRadius={12}
            elevation={6}
            onPress={() => flatListRef.current?.scrollToEnd({animated: true})}
            pressStyle={{opacity: 0.7, scale: 0.9}}
            accessible accessibilityRole="button" accessibilityLabel="Jump to bottom">
            <Icon name="arrow-down" size={18} color="$color9" />
          </YStack>
        )}
      </KeyboardAvoidingView>

      <ChatDrawer
        visible={a.showDrawer}
        onClose={() => a.setShowDrawer(false)}
        sortedActiveSessions={a.sortedActiveSessions}
        archivedSessions={a.archivedSessions}
        sessionLabels={a.sessionLabels}
        allLabels={a.allLabels}
        labelFilter={a.labelFilter}
        setLabelFilter={a.setLabelFilter}
        showArchived={a.showArchived}
        setShowArchived={a.setShowArchived}
        labelInput={a.labelInput}
        setLabelInput={a.setLabelInput}
        starredIds={a.starredIds}
        setStarredIds={a.setStarredIds}
        setSessionLabels={a.setSessionLabels}
        setAllLabels={a.setAllLabels}
      />

      <ChatBottomSheets
        showInfo={a.showInfo}
        setShowInfo={a.setShowInfo}
        showSearchSessions={a.showSearchSessions}
        setShowSearchSessions={a.setShowSearchSessions}
        handleSelectSearchSession={a.handleSelectSearchSession}
        showSoulPicker={a.showSoulPicker}
        setShowSoulPicker={a.setShowSoulPicker}
        showSettings={a.showSettings}
        setShowSettings={a.setShowSettings}
        showChatSettings={a.showChatSettings}
        setShowChatSettings={a.setShowChatSettings}
        showSystemPrompt={a.showSystemPrompt}
        setShowSystemPrompt={a.setShowSystemPrompt}
        forwardTo={a.forwardTo}
        setForwardTo={a.setForwardTo}
        safeSessions={a.safeSessions}
        activeSessionId={a.activeSessionId}
        messages={a.messages}
        currentSoul={a.currentSoul}
        souls={a.souls}
        switchSoul={a.switchSoul}
        isConnected={a.isConnected}
        chatBackground={a.chatBackground}
        updateTheme={a.updateTheme}
        themeMode={a.themeMode}
        sessionLabels={a.sessionLabels}
        setSessionLabels={a.setSessionLabels}
        allLabels={a.allLabels}
        setAllLabels={a.setAllLabels}
        labelInput={a.labelInput}
        setLabelInput={a.setLabelInput}
        forwardMessage={a.forwardMessage}
        createSession={a.createSession}
        handleExportChat={a.handleExportChat}
        setShowSearch={a.setShowSearch}
      />
    </SafeAreaView>
  );
}
