import React, {useEffect, useState, useCallback, useRef} from 'react';
import {
  FlatList,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  Modal,
  TextInput as RNTextInput,
  Alert,
  Keyboard,
  Share,
  RefreshControl,
  useColorScheme,
} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {YStack, XStack, Text, useTheme} from 'tamagui';
import {useChatStore} from '../stores/chat-store';
import {useModelStore} from '../stores/model-store';
import {useOnlineStatus} from '../hooks/useOnlineStatus';
import {useHybridStore} from '../stores/hybrid-inference-store';
import {MessageBubble} from '../components/MessageBubble';
import {ChatInput} from '../components/ChatInput';
import {ReasoningPanel} from '../components/ReasoningPanel';
import {SearchSessionsModal} from '../components/SearchSessionsModal';
import {StatusBadge} from '../components/StatusBadge';
import {triggerHaptic} from '../services/haptics';
import {api} from '../services/api-client';
import {pickImage, imageDataUrl} from '../services/image-upload';
import {startRecording, transcribeAudio} from '../services/voice-input';
import {toast} from '../services/toast';
import {useSettingsStore} from '../stores/settings-store';
import * as pinsService from '../services/pins';
import * as starsService from '../services/stars';
import * as labelsService from '../services/labels';
import {getCachedActiveSessionId} from '../services/offline-cache';
import type {Message, Session} from '../types';
import {Icon} from '../components/Icon';

const SUGGESTIONS = [
  'Tell me something interesting',
  'Help me brainstorm',
  'Explain a concept',
];

const BG_PRESETS = [
  {label: 'None', value: ''},
  {label: 'Navy', value: '#1a1a2e'},
  {label: 'Plum', value: '#2d1b2e'},
  {label: 'Forest', value: '#1b2e1a'},
  {label: 'Amber', value: '#2e2e1b'},
  {label: 'Maroon', value: '#2a1a1a'},
  {label: 'Cream', value: '#f5f0e8'},
  {label: 'Ice', value: '#e8f0f5'},
  {label: 'Lavender', value: '#f0e8f5'},
  {label: 'Rose', value: '#f5e8e8'},
];

export function ChatScreen() {
  const theme = useTheme();
  const {
    sessions,
    activeSessionId,
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
    archiveSession,
    renameSession,
    deleteMessage,
    forwardMessage,
    createSession,
    offlineQueue,
    retryPendingSends,
  } = useChatStore();
  const {health, currentSoul, souls, switchSoul} = useModelStore();
  const isDark = useColorScheme() === 'dark';
  const themeMode = useSettingsStore(s => s.theme);
  const updateTheme = useSettingsStore(s => s.update);
  const chatBackground = useSettingsStore(s => s.chatBackground);
  const online = useOnlineStatus();
  const hybrid = useHybridStore();
  const flatListRef = useRef<FlatList>(null);
  const lastHeaderTap = useRef<number>(0);
  const [atBottom, setAtBottom] = useState(true);
  const [showDrawer, setShowDrawer] = useState(false);
  const [showSoulPicker, setShowSoulPicker] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [showSearch, setShowSearch] = useState(false);
  const [showSearchSessions, setShowSearchSessions] = useState(false);
  const [showInfo, setShowInfo] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [refreshing, setRefreshing] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [editingMessage, setEditingMessage] = useState<string | null>(null);
  const [selectMode, setSelectMode] = useState(false);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [replyTo, setReplyTo] = useState<Message | null>(null);
  const [forwardTo, setForwardTo] = useState<Message | null>(null);
  const [showArchived, setShowArchived] = useState(false);
  const [labelFilter, setLabelFilter] = useState<string | null>(null);
  const [sessionLabels, setSessionLabels] = useState<Record<string, string[]>>({});
  const [allLabels, setAllLabels] = useState<string[]>([]);
  const [labelInput, setLabelInput] = useState('');
  const [starredIds, setStarredIds] = useState<string[]>([]);
  const [pinnedIds, setPinnedIds] = useState<string[]>([]);
  const safeSessions = sessions ?? [];
  const activeSessions = safeSessions.filter(s => !s.archived);
  const archivedSessions = safeSessions.filter(s => s.archived);
  const recordingStopRef = useRef<(() => Promise<{uri: string; duration: number} | null>) | null>(null);
  const [voiceMessageMode, setVoiceMessageMode] = useState(false);
  const voiceTimerRef = useRef<{start: number} | null>(null);

  useEffect(() => {
    if (activeSessionId) {
      pinsService.getPinnedIds(activeSessionId).then(setPinnedIds);
    } else {
      setPinnedIds([]);
    }
  }, [activeSessionId]);

  useEffect(() => {
    refreshSessions();
    starsService.getStarredIds().then(setStarredIds);
    labelsService.getAllDistinctLabels().then(setAllLabels);
    (async () => {
      const cachedId = await getCachedActiveSessionId();
      if (cachedId && !useChatStore.getState().activeSessionId) {
        await loadSession(cachedId);
      }
    })();
  }, []);

  useEffect(() => {
    if (showInfo && activeSessionId) {
      labelsService.getLabels(activeSessionId).then(labels => {
        setSessionLabels(prev => ({...prev, [activeSessionId]: labels}));
      });
    }
  }, [showInfo, activeSessionId]);

  useEffect(() => {
    if (showDrawer) {
      (async () => {
        const all: Record<string, string[]> = {};
        for (const s of safeSessions) {
          all[s.id] = await labelsService.getLabels(s.id);
        }
        setSessionLabels(prev => ({...prev, ...all}));
        const distinct = await labelsService.getAllDistinctLabels();
        setAllLabels(distinct);
      })();
    }
  }, [showDrawer, sessions]);

  // Sort sessions: starred first (by star order), then by updated_at
  const sortedActiveSessions = [...activeSessions].sort((a, b) => {
    const aStarred = starredIds.includes(a.id);
    const bStarred = starredIds.includes(b.id);
    if (aStarred && !bStarred) return -1;
    if (!aStarred && bStarred) return 1;
    if (aStarred && bStarred) {
      return starredIds.indexOf(a.id) - starredIds.indexOf(b.id);
    }
    return new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime();
  });

  // Keyboard shortcuts for external keyboards
  useEffect(() => {
    const {Keyboard} = require('react-native');
    const sub = Keyboard.addListener('keyboardDidHide', () => {
      // Escape behavior — dismiss any open modal
      if (showDrawer) setShowDrawer(false);
      else if (showSoulPicker) setShowSoulPicker(false);
      else if (showSettings) setShowSettings(false);
      else if (showSearch) setShowSearch(false);
      else if (showInfo) setShowInfo(false);
    });
    return () => sub.remove();
  }, [showDrawer, showSoulPicker, showSettings, showSearch, showInfo]);

  const toggleSelectMode = useCallback(() => {
    setSelectMode(s => !s);
    setSelectedIds(new Set());
  }, []);

  const toggleSelectMessage = useCallback((id: string) => {
    setSelectedIds(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const deleteSelected = useCallback(async () => {
    for (const id of selectedIds) deleteMessage(id);
    setSelectedIds(new Set());
    setSelectMode(false);
    triggerHaptic('medium');
  }, [selectedIds, deleteMessage]);

  const onPullRefresh = useCallback(async () => {
    setRefreshing(true);
    await triggerHaptic('light');
    await refreshSessions();
    const state = useChatStore.getState();
    if (state.activeSessionId) {
      await loadSession(state.activeSessionId);
    }
    if (state.offlineQueue > 0) {
      await retryPendingSends();
    }
    setRefreshing(false);
  }, [refreshSessions, loadSession, retryPendingSends]);

  useEffect(() => {
    if (atBottom && messages.length > 0) {
      setTimeout(() => {
        flatListRef.current?.scrollToEnd({animated: true});
      }, 50);
    }
  }, [messages, atBottom]);

  const handleFile = useCallback(
    (content: string, name: string) => {
      const truncated = content.length > 2000 ? content.slice(0, 2000) + '\n...(truncated)' : content;
      sendMessage(`Here's the file "${name}":\n\n${truncated}`);
    },
    [sendMessage],
  );

  const handleSend = useCallback(
    (text: string) => {
      sendMessage(text);
    },
    [sendMessage],
  );

  const handleImage = useCallback(async () => {
    try {
      const result = await pickImage();
      if (result) {
        const dataUrl = imageDataUrl(result);
        sendMessage('What do you see in this image?', [dataUrl]);
      }
    } catch (e: any) {
      toast.error(e.message || 'Failed to pick image');
    }
  }, [sendMessage]);

  const handleVoice = useCallback(async () => {
    if (isRecording) {
      // Stop recording
      const stop = recordingStopRef.current;
      if (stop) {
        const recording = await stop();
        setIsRecording(false);
        recordingStopRef.current = null;
        if (recording) {
          if (voiceMessageMode) {
            // Send raw audio
            let sid = activeSessionId;
            if (!sid) {
              await createSession();
              sid = useChatStore.getState().activeSessionId;
            }
            if (sid) {
              try {
                const result = await api.sendVoiceMessage(sid, recording.uri, recording.duration);
                if (result.message_id) {
                  const voiceMsg: Message = {
                    id: Date.now().toString(36) + Math.random().toString(36).slice(2, 5),
                    role: 'user',
                    content: '🎤 Voice message',
                    timestamp: Date.now(),
                    audio_path: result.audio_path,
                    audio_duration_ms: recording.duration,
                    _voice: true,
                    status: 'sent',
                  };
                  useChatStore.setState(s => ({
                    messages: [...s.messages, voiceMsg],
                  }));
                  toast.success('Voice message sent');
                }
              } catch {
                toast.error('Failed to send voice message');
              }
            }
          } else {
            // Transcribe and send as text
            try {
              const text = await transcribeAudio(recording.uri);
              if (text) {
                sendMessage(text);
              } else {
                toast.warn('Could not transcribe audio');
              }
            } catch {
              toast.error('Transcription failed');
            }
          }
        }
      }
    } else {
      // Start recording
      try {
        const {stop} = await startRecording();
        recordingStopRef.current = stop;
        setIsRecording(true);
        triggerHaptic('medium');
      } catch (e: any) {
        toast.error(e.message || 'Failed to start recording');
      }
    }
  }, [isRecording, sendMessage, voiceMessageMode, activeSessionId, createSession]);

  const handleSuggestion = useCallback(
    (s: string) => {
      sendMessage(s);
    },
    [sendMessage],
  );

  const handleSelectSearchSession = useCallback(
    (sessionId: string) => {
      loadSession(sessionId);
    },
    [loadSession],
  );

  const handleExportChat = useCallback(async () => {
    if (!activeSessionId || messages.length === 0) {
      toast.info('Nothing to export');
      return;
    }
    const lines = messages.map(m => {
      const role = m.role === 'user' ? 'You' : 'Assistant';
      return `**${role}:** ${m.content}`;
    });
    const md = `# Conversation\n\n${lines.join('\n\n')}`;
    try {
      await Share.share({message: md, title: 'Conversation'});
      toast.success('Exported');
    } catch {}
  }, [activeSessionId, messages]);

  const renderItem = useCallback(
    ({item}: {item: Message}) => (
      <MessageBubble
        message={item}
        sessionId={activeSessionId || undefined}
        highlight={searchQuery ? item.content.toLowerCase().includes(searchQuery.toLowerCase()) : false}
        onRegenerate={
          item.role === 'assistant' ? () => regenerate(item.id) : undefined
        }
        onFeedback={
          item.role === 'assistant'
            ? positive => recordFeedback(item.id, positive)
            : undefined
        }
        onDelete={() => deleteMessage(item.id)}
        onEdit={
          item.role === 'user' ? (newContent: string) => setEditingMessage(newContent) : undefined
        }
        onReply={() => setReplyTo(item)}
        onForward={() => setForwardTo(item)}
        selectMode={selectMode}
        selected={selectedIds.has(item.id)}
        onSelect={() => toggleSelectMessage(item.id)}
        onLongPressSelect={() => {
          if (!selectMode) {
            setSelectMode(true);
            setSelectedIds(new Set([item.id]));
          }
        }}
      />
    ),
    [regenerate, recordFeedback, searchQuery, deleteMessage, activeSessionId, selectMode, selectedIds, toggleSelectMessage],
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
    <SafeAreaView style={{flex: 1, backgroundColor: 'var(--background)'}} edges={['top']}>
      <KeyboardAvoidingView
        style={{flex: 1}}
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
        keyboardVerticalOffset={0}>
        {/* Header — sleek minimal */}
        <XStack
          paddingHorizontal={16}
          paddingVertical={12}
          borderBottomWidth={0.5}
          borderBottomColor="$borderColor"
          backgroundColor="$background"
          alignItems="center"
          justifyContent="space-between"
          opacity={0.98}>
          {/* Left: menu + title */}
          <XStack alignItems="center" gap={12}>
            <YStack
              width={36} height={36} borderRadius={12}
              alignItems="center" justifyContent="center"
              onPress={() => setShowDrawer(true)}
              pressStyle={{opacity: 0.6, scale: 0.95}}>
              <Icon name="menu" size={18} color={(theme.color11?.val || '#6B7280')} />
            </YStack>
            <YStack
              alignItems="flex-start"
              onPress={() => {
                const now = Date.now();
                if (now - lastHeaderTap.current < 300) {
                  flatListRef.current?.scrollToOffset({offset: 0, animated: true});
                }
                lastHeaderTap.current = now;
              }}>
              <XStack alignItems="center" gap={6}>
                <Text fontSize={17} fontWeight="700" letterSpacing={-0.3} color="$color">Chat</Text>
                {currentSoul && (
                  <YStack
                    backgroundColor="rgba(124, 82, 196, 0.1)"
                    paddingHorizontal={10}
                    paddingVertical={3}
                    borderRadius={999}
                    borderWidth={0.5}
                    borderColor="rgba(124, 82, 196, 0.18)"
                    onPress={() => setShowSoulPicker(true)}
                    pressStyle={{opacity: 0.7, scale: 0.95}}>
                    <Text fontSize={11} fontWeight="600" color="$color9">{currentSoul.name}</Text>
                  </YStack>
                )}
              </XStack>
            </YStack>
          </XStack>

          {/* Right: single overflow menu */}
          <YStack
            width={36} height={36} borderRadius={12}
            alignItems="center" justifyContent="center"
            onPress={() => setShowSettings(true)}
            pressStyle={{opacity: 0.6, scale: 0.95}}>
            <Icon name="more-vertical" size={18} color={(theme.color11?.val || '#6B7280')} />
          </YStack>
        </XStack>

        {error && (
          <XStack
            paddingHorizontal={16} paddingVertical={10}
            backgroundColor="rgba(239, 68, 68, 0.06)"
            borderBottomWidth={0.5} borderBottomColor="rgba(239, 68, 68, 0.12)"
            alignItems="center" gap={8}
            onPress={clearError}>
            <YStack width={6} height={6} borderRadius={3} backgroundColor="#EF4444" />
            <Text fontSize={12} color="#EF4444" flex={1} numberOfLines={2}>{error}</Text>
            <Icon name="x" size={14} color="#EF4444" />
          </XStack>
        )}

        {!online && (
          <XStack
            paddingHorizontal={16} paddingVertical={8}
            borderBottomWidth={0.5} borderBottomColor="$borderColor"
            alignItems="center" gap={6}>
            <YStack width={5} height={5} borderRadius={3} backgroundColor="#F59E0B" />
            <Text fontSize={11} fontWeight="500" color="$color10">Offline</Text>
            {offlineQueue > 0 && (
              <YStack onPress={retryPendingSends}>
                <Text fontSize={11} color="$color11" textDecorationLine="underline">{offlineQueue} queued</Text>
              </YStack>
            )}
          </XStack>
        )}

        {showSearch && (
          <XStack
            paddingHorizontal={16} paddingVertical={8}
            gap={8} alignItems="center">
            <RNTextInput
              style={{
                flex: 1, fontSize: 13,
                color: (theme.color?.val || '#111827'),
                backgroundColor: 'rgba(124, 82, 196, 0.06)',
                borderRadius: 10,
                paddingHorizontal: 12, paddingVertical: 7,
                borderWidth: 0.5, borderColor: (theme.borderColor?.val || '#E5E7EB'),
              }}
              value={searchQuery}
              onChangeText={setSearchQuery}
              placeholder="Search messages..."
              placeholderTextColor={(theme.color10?.val || '#9CA3AF')}
              autoFocus
            />
            <YStack
              width={28} height={28} borderRadius={9}
              alignItems="center" justifyContent="center"
              backgroundColor="rgba(124, 82, 196, 0.06)"
              onPress={() => { setShowSearch(false); setSearchQuery(''); }}
              pressStyle={{opacity: 0.6}}>
              <Icon name="x" size={14} color={(theme.color10?.val || '#9CA3AF')} />
            </YStack>
          </XStack>
        )}

        {messages.length === 0 && !streaming ? (
          <YStack
            flex={1} alignItems="center" justifyContent="center"
            paddingHorizontal={32}
            backgroundColor={chatBackground || 'var(--background)'}>
            {/* Gradient soul avatar — glass ring */}
            <YStack
              width={80} height={80} borderRadius={40}
              backgroundColor="rgba(124, 82, 196, 0.08)"
              alignItems="center" justifyContent="center"
              marginBottom={24}
              borderWidth={0.5}
              borderColor="rgba(124, 82, 196, 0.15)">
              <YStack
                width={56} height={56} borderRadius={28}
                backgroundColor="rgba(124, 82, 196, 0.1)"
                alignItems="center" justifyContent="center">
                <Icon name="message-circle" size={26} color={(theme.color9?.val || '#7C52C4')} />
              </YStack>
            </YStack>

            {/* Title */}
            <Text fontSize={22} fontWeight="700" letterSpacing={-0.5} color="$color" marginBottom={8} textAlign="center">
              {currentSoul ? currentSoul.name : 'Chat'}
            </Text>

            {/* Subtitle */}
            <Text fontSize={14} color="$color11" textAlign="center" marginBottom={36} lineHeight={20} maxWidth={260}>
              {currentSoul
                ? (currentSoul.description || 'Ask me anything')
                : 'Start a conversation'}
            </Text>

            {/* Suggestion chips — horizontal, pill-style */}
            <XStack gap={10} flexWrap="wrap" justifyContent="center">
              {SUGGESTIONS.map(s => (
                <YStack
                  key={s}
                  paddingHorizontal={18} paddingVertical={10}
                  borderRadius={999}
                  backgroundColor="rgba(124, 82, 196, 0.06)"
                  borderWidth={0.5}
                  borderColor="rgba(124, 82, 196, 0.12)"
                  onPress={() => handleSuggestion(s)}
                  pressStyle={{opacity: 0.7, scale: 0.96}}>
                  <Text fontSize={13} fontWeight="500" color="$color9">{s}</Text>
                </YStack>
              ))}
            </XStack>

            {/* Hint */}
            <Text fontSize={11} color="$color10" marginTop={32} textAlign="center" letterSpacing={0.3}>
              Swipe left on a message to delete it
            </Text>
          </YStack>
        ) : (
          <Pressable
            style={{flex: 1, backgroundColor: chatBackground || 'var(--background)'}}
            onPress={() => Keyboard.dismiss()}>
            <FlatList
              ref={flatListRef}
              data={messages}
              renderItem={renderItem}
              keyExtractor={keyExtractor}
              contentContainerStyle={{paddingTop: 8, paddingBottom: 8}}
              onScroll={onScroll}
              scrollEventThrottle={16}
              refreshControl={
                <RefreshControl
                  refreshing={refreshing}
                  onRefresh={onPullRefresh}
                  tintColor={(theme.color9?.val || '#7C52C4')}
                />
              }
              onContentSizeChange={() => {
              if (atBottom) {
                flatListRef.current?.scrollToEnd({animated: false});
              }
            }}
            />
          </Pressable>
        )}

        {/* Pinned messages banner */}
        {pinnedIds.length > 0 && !selectMode && (
          <XStack
            paddingHorizontal={16} paddingVertical={8}
            gap={8} alignItems="center"
            onPress={() => {
              const firstPinnedIndex = messages.findIndex(m => pinnedIds.includes(m.id));
              if (firstPinnedIndex >= 0) {
                flatListRef.current?.scrollToIndex({index: firstPinnedIndex, animated: true, viewPosition: 0});
              }
            }}
            pressStyle={{opacity: 0.7}}>
            <YStack width={24} height={24} borderRadius={8} backgroundColor="rgba(124, 82, 196, 0.08)" alignItems="center" justifyContent="center">
              <Icon name="pin" size={12} color="$color9" />
            </YStack>
            <Text fontSize={11} fontWeight="500" color="$color9">
              {pinnedIds.length} pinned {pinnedIds.length === 1 ? 'message' : 'messages'}
            </Text>
          </XStack>
        )}

        {/* Select mode toolbar */}
        {selectMode && (
          <XStack
            paddingHorizontal={16} paddingVertical={10}
            borderTopWidth={0.5} borderTopColor="$borderColor"
            alignItems="center" justifyContent="space-between"
            backgroundColor="rgba(124, 82, 196, 0.04)">
            <YStack onPress={() => {
              if (selectedIds.size === messages.length) {
                setSelectedIds(new Set());
              } else {
                setSelectedIds(new Set(messages.map(m => m.id)));
              }
            }} pressStyle={{opacity: 0.6}}>
              <Text fontSize={13} fontWeight="600" color="$color9">
                {selectedIds.size === messages.length ? 'Deselect all' : 'Select all'}
              </Text>
            </YStack>
            <XStack gap={12} alignItems="center">
              <Text fontSize={12} color="$color10">{selectedIds.size} selected</Text>
              <YStack paddingHorizontal={10} paddingVertical={5} borderRadius={8}
                backgroundColor={selectedIds.size > 0 ? 'rgba(239, 68, 68, 0.1)' : 'transparent'}
                opacity={selectedIds.size === 0 ? 0.4 : 1}
                onPress={deleteSelected}
                disabled={selectedIds.size === 0}>
                <Text fontSize={13} fontWeight="600"
                  color={selectedIds.size > 0 ? '#EF4444' : '$color10'}>Delete</Text>
              </YStack>
              <YStack paddingHorizontal={10} paddingVertical={5} borderRadius={8}
                backgroundColor="rgba(124, 82, 196, 0.08)" onPress={toggleSelectMode}>
                <Text fontSize={13} fontWeight="600" color="$color9">Done</Text>
              </YStack>
            </XStack>
          </XStack>
        )}

        <ReasoningPanel visible={streaming && messages.length > 0 && !messages[messages.length - 1]?.content} />

        {/* Reply quote bar */}
        {replyTo && (
          <XStack
            paddingHorizontal={16} paddingVertical={8}
            gap={10} alignItems="center"
            backgroundColor="rgba(124, 82, 196, 0.04)"
            borderTopWidth={0.5} borderTopColor="$borderColor">
            <YStack width={3} height={32} borderRadius={2} backgroundColor="$color9" />
            <YStack flex={1}>
              <Text fontSize={11} fontWeight="600" color="$color9" marginBottom={1}>
                Replying to {replyTo.role === 'user' ? 'yourself' : 'assistant'}
              </Text>
              <Text fontSize={12} color="$color10" numberOfLines={1}>
                {replyTo.content}
              </Text>
            </YStack>
            <YStack
              width={24} height={24} borderRadius={8}
              backgroundColor="rgba(124, 82, 196, 0.06)"
              alignItems="center" justifyContent="center"
              onPress={() => setReplyTo(null)}
              pressStyle={{opacity: 0.6}}>
              <Icon name="x" size={12} color="$color10" />
            </YStack>
          </XStack>
        )}

        {/* Forward to session — bottom sheet */}
        <Modal
          visible={forwardTo !== null}
          transparent
          animationType="slide"
          onRequestClose={() => setForwardTo(null)}>
          <YStack flex={1} justifyContent="flex-end">
            <YStack flex={1} backgroundColor="rgba(0,0,0,0.3)" onPress={() => setForwardTo(null)} />
            <YStack
              backgroundColor="$background"
              borderTopLeftRadius={24}
              borderTopRightRadius={24}
              maxHeight="65%"
              overflow="hidden">
              <YStack alignItems="center" paddingTop={12} paddingBottom={2}>
                <YStack width={40} height={5} borderRadius={3} backgroundColor="$borderColor" opacity={0.4} />
              </YStack>

              <XStack
                paddingHorizontal={20} paddingVertical={14}
                borderBottomWidth={0.5} borderBottomColor="$borderColor"
                alignItems="center" justifyContent="space-between">
                <Text fontSize={17} fontWeight="700" letterSpacing={-0.3} color="$color">Forward to...</Text>
                <YStack width={28} height={28} borderRadius={9} alignItems="center" justifyContent="center"
                  onPress={() => setForwardTo(null)} pressStyle={{opacity: 0.6}}>
                  <Icon name="x" size={14} color={(theme.color11?.val || '#6B7280')} />
                </YStack>
              </XStack>

              {forwardTo && (
                <YStack
                  marginHorizontal={16} marginTop={10} marginBottom={6}
                  paddingHorizontal={14} paddingVertical={10}
                  borderRadius={12}
                  backgroundColor="rgba(124, 82, 196, 0.04)"
                  borderWidth={0.5} borderColor="rgba(124, 82, 196, 0.12)">
                  <Text fontSize={11} fontWeight="600" color="$color9" marginBottom={2}>MESSAGE</Text>
                  <Text fontSize={13} color="$color10" numberOfLines={2}>{forwardTo.content}</Text>
                </YStack>
              )}

              <FlatList
                data={safeSessions}
                keyExtractor={s => s.id}
                contentContainerStyle={{paddingHorizontal: 16, paddingVertical: 4}}
                renderItem={({item: session}) => (
                  <XStack
                    paddingVertical={12} paddingHorizontal={14}
                    marginVertical={2}
                    borderRadius={12}
                    alignItems="center" gap={12}
                    onPress={async () => {
                      if (forwardTo) {
                        await forwardMessage(forwardTo.content, session.id);
                      }
                      setForwardTo(null);
                    }}
                    pressStyle={{backgroundColor: 'rgba(124, 82, 196, 0.06)', scale: 0.98}}>
                    <YStack width={36} height={36} borderRadius={12}
                      backgroundColor="rgba(124, 82, 196, 0.08)" alignItems="center" justifyContent="center">
                      <Icon name="message-circle" size={16} color="$color9" />
                    </YStack>
                    <YStack flex={1}>
                      <Text fontSize={14} fontWeight="500" numberOfLines={1} color="$color">
                        {session.name || 'New conversation'}
                      </Text>
                      <Text fontSize={11} color="$color10" marginTop={1}>
                        {session.message_count || 0} messages
                      </Text>
                    </YStack>
                  </XStack>
                )}
                ListEmptyComponent={
                  <YStack padding={40} alignItems="center">
                    <Text fontSize={13} color="$color10">No conversations</Text>
                  </YStack>
                }
              />
              <YStack height={Platform.OS === 'ios' ? 34 : 16} />
            </YStack>
          </YStack>
        </Modal>

        <ChatInput
          onSend={handleSend}
          onImage={handleImage}
          onVoice={handleVoice}
          onFile={handleFile}
          disabled={streaming}
          onStop={cancelStream}
          isRecording={isRecording}
          sessionId={activeSessionId}
          editText={editingMessage}
          onCancelEdit={() => setEditingMessage(null)}
          voiceMessageMode={voiceMessageMode}
          onVoiceMessageToggle={() => setVoiceMessageMode(v => !v)}
        />

        {!atBottom && messages.length > 0 && (
          <YStack
            position="absolute"
            right={20}
            bottom={80}
            width={40}
            height={40}
            borderRadius={20}
            backgroundColor="rgba(124, 82, 196, 0.15)"
            borderWidth={0.5}
            borderColor="rgba(124, 82, 196, 0.25)"
            alignItems="center"
            justifyContent="center"
            shadowColor="$color9"
            shadowOffset={{width: 0, height: 4}}
            shadowOpacity={0.25}
            shadowRadius={12}
            elevation={6}
            onPress={() => flatListRef.current?.scrollToEnd({animated: true})}
            pressStyle={{opacity: 0.7, scale: 0.9}}>
            <Icon name="arrow-down" size={18} color="$color9" />
          </YStack>
        )}
      </KeyboardAvoidingView>

      {/* Conversation Info — bottom sheet */}
      <Modal visible={showInfo} animationType="slide" transparent onRequestClose={() => setShowInfo(false)}>
        <YStack flex={1} justifyContent="flex-end">
          <YStack
            flex={1}
            backgroundColor="rgba(0,0,0,0.3)"
            onPress={() => setShowInfo(false)}
          />
          <YStack
            backgroundColor="$background"
            borderTopLeftRadius={24}
            borderTopRightRadius={24}
            maxHeight="85%"
            overflow="hidden">
            {/* Handle bar */}
            <YStack alignItems="center" paddingTop={12} paddingBottom={2}>
              <YStack width={40} height={5} borderRadius={3} backgroundColor="$borderColor" opacity={0.4} />
            </YStack>

            <XStack
              paddingHorizontal={20} paddingVertical={14}
              borderBottomWidth={0.5} borderBottomColor="$borderColor"
              alignItems="center" justifyContent="space-between">
              <Text fontSize={17} fontWeight="700" letterSpacing={-0.3} color="$color">Details</Text>
              <YStack
                width={28} height={28} borderRadius={9}
                alignItems="center" justifyContent="center"
                onPress={() => setShowInfo(false)}
                pressStyle={{opacity: 0.6}}>
                <Icon name="x" size={14} color={(theme.color11?.val || '#6B7280')} />
              </YStack>
            </XStack>

            {/* Stats grid */}
            <XStack paddingHorizontal={20} paddingVertical={16} gap={10}>
              {[
                {label: 'Messages', value: String(messages.length)},
                {label: 'Words', value: String(messages.reduce((sum, m) => sum + (m.content?.split(/\s+/).length || 0), 0))},
                {label: 'Characters', value: String(messages.reduce((sum, m) => sum + (m.content?.length || 0), 0))},
              ].map(stat => (
                <YStack key={stat.label} flex={1} paddingVertical={14} paddingHorizontal={10} borderRadius={14}
                  backgroundColor="rgba(124, 82, 196, 0.04)" alignItems="center">
                  <Text fontSize={20} fontWeight="700" color="$color9">{stat.value}</Text>
                  <Text fontSize={10} fontWeight="500" color="$color10" marginTop={4}>{stat.label}</Text>
                </YStack>
              ))}
            </XStack>

            {/* Meta rows */}
            <YStack paddingHorizontal={20} paddingBottom={4}>
              <XStack justifyContent="space-between" paddingVertical={12}
                borderBottomWidth={0.5} borderBottomColor="$borderColor">
                <Text fontSize={13} color="$color11">Session</Text>
                <Text fontSize={13} color="$color" fontWeight="500" numberOfLines={1}>
                  {activeSessionId?.slice(0, 12) || 'None'}
                </Text>
              </XStack>
              {currentSoul && (
                <XStack justifyContent="space-between" paddingVertical={12}
                  borderBottomWidth={0.5} borderBottomColor="$borderColor">
                  <Text fontSize={13} color="$color11">Soul</Text>
                  <Text fontSize={13} color="$color" fontWeight="500">{currentSoul.name}</Text>
                </XStack>
              )}
              <XStack justifyContent="space-between" paddingVertical={12}
                borderBottomWidth={0.5} borderBottomColor="$borderColor">
                <Text fontSize={13} color="$color11">Status</Text>
                <XStack alignItems="center" gap={6}>
                  <YStack width={7} height={7} borderRadius={4}
                    backgroundColor={isConnected ? '#22C55E' : '#EF4444'} />
                  <Text fontSize={13} fontWeight="500" color="$color">
                    {isConnected ? 'Connected' : 'Offline'}
                  </Text>
                </XStack>
              </XStack>
            </YStack>

            {/* Chat Background */}
            <YStack paddingHorizontal={20} paddingTop={12} paddingBottom={8}>
              <Text fontSize={11} fontWeight="700" letterSpacing={0.6} color="$color10" marginBottom={12}>CHAT BACKGROUND</Text>
              <XStack flexWrap="wrap" gap={10}>
                {BG_PRESETS.map(p => {
                  const active = chatBackground === p.value;
                  return (
                    <YStack
                      key={p.value || 'none'}
                      width={36} height={36} borderRadius={12}
                      justifyContent="center" alignItems="center"
                      borderWidth={2}
                      borderColor={active ? '$color9' : 'transparent'}
                      backgroundColor={p.value ? p.value : (theme.background?.val || '#FFFFFF')}
                      style={!p.value ? {borderColor: (theme.borderColor?.val || '#E5E7EB')} : {}}
                      onPress={() => updateTheme({chatBackground: p.value})}
                      pressStyle={{scale: 0.88}}>
                      {active && (
                        <Icon name="check" size={12} color="white" />
                      )}
                    </YStack>
                  );
                })}
              </XStack>
            </YStack>

            {/* Labels */}
            <YStack paddingHorizontal={20} paddingTop={16} paddingBottom={24}>
              <Text fontSize={11} fontWeight="700" letterSpacing={0.6} color="$color10" marginBottom={12}>LABELS</Text>
              {activeSessionId && (sessionLabels[activeSessionId] || []).length > 0 && (
                <XStack flexWrap="wrap" gap={6} marginBottom={12}>
                  {(sessionLabels[activeSessionId] || []).map(label => (
                    <YStack
                      key={label}
                      flexDirection="row" alignItems="center" gap={4}
                      backgroundColor="rgba(124, 82, 196, 0.08)"
                      paddingHorizontal={10} paddingVertical={5} borderRadius={999}
                      onPress={async () => {
                        await labelsService.removeLabel(activeSessionId, label);
                        const labels = await labelsService.getLabels(activeSessionId);
                        setSessionLabels(prev => ({...prev, [activeSessionId]: labels}));
                        const distinct = await labelsService.getAllDistinctLabels();
                        setAllLabels(distinct);
                      }}>
                      <Text fontSize={12} fontWeight="500" color="$color9">{label}</Text>
                      <Icon name="x" size={10} color="$color9" />
                    </YStack>
                  ))}
                </XStack>
              )}
              <XStack gap={10}>
                <RNTextInput
                  style={{
                    flex: 1, fontSize: 13,
                    color: (theme.color?.val || '#111827'),
                    backgroundColor: 'rgba(124, 82, 196, 0.04)',
                    borderRadius: 12,
                    paddingHorizontal: 12, paddingVertical: 9,
                    borderWidth: 0.5, borderColor: (theme.borderColor?.val || '#E5E7EB'),
                  }}
                  value={labelInput}
                  onChangeText={setLabelInput}
                  placeholder="Add a label..."
                  placeholderTextColor={(theme.color10?.val || '#9CA3AF')}
                  returnKeyType="done"
                  onSubmitEditing={async () => {
                    if (labelInput.trim() && activeSessionId) {
                      await labelsService.addLabel(activeSessionId, labelInput.trim());
                      setLabelInput('');
                      const labels = await labelsService.getLabels(activeSessionId);
                      setSessionLabels(prev => ({...prev, [activeSessionId]: labels}));
                      const distinct = await labelsService.getAllDistinctLabels();
                      setAllLabels(distinct);
                    }
                  }}
                />
              </XStack>
            </YStack>
          </YStack>
        </YStack>
      </Modal>

      {/* Session drawer — bottom sheet */}
      <Modal visible={showDrawer} animationType="slide" transparent>
        <YStack flex={1} justifyContent="flex-end">
          <YStack
            flex={1}
            backgroundColor="rgba(0,0,0,0.3)"
            onPress={() => setShowDrawer(false)}
          />
          <YStack
            backgroundColor="$background"
            borderTopLeftRadius={24}
            borderTopRightRadius={24}
            maxHeight="85%"
            overflow="hidden">
            {/* Handle bar */}
            <YStack alignItems="center" paddingTop={10} paddingBottom={4}>
              <YStack width={40} height={5} borderRadius={3} backgroundColor="$borderColor" opacity={0.5} />
            </YStack>

            <XStack
              paddingHorizontal={20} paddingVertical={14}
              paddingBottom={12}
              borderBottomWidth={0.5} borderBottomColor="$borderColor"
              alignItems="center" justifyContent="space-between">
              <Text fontSize={17} fontWeight="700" letterSpacing={-0.3} color="$color">Conversations</Text>
              <YStack
                width={32} height={32} borderRadius={10}
                alignItems="center" justifyContent="center"
                onPress={() => setShowDrawer(false)}
                pressStyle={{opacity: 0.6}}>
                <Icon name="x" size={16} color={(theme.color11?.val || '#6B7280')} />
              </YStack>
            </XStack>

            {/* Session list */}
            <FlatList
              data={labelFilter ? sortedActiveSessions.filter(s => (sessionLabels[s.id] || []).includes(labelFilter!)) : sortedActiveSessions}
              keyExtractor={item => item.id}
              renderItem={({item: session}) => {
                const isStarred = starredIds.includes(session.id);
                const isActive = session.id === useChatStore.getState().activeSessionId;
                return (
                <XStack
                  paddingHorizontal={16} paddingVertical={12}
                  marginHorizontal={12} marginVertical={2}
                  borderRadius={12}
                  alignItems="center" justifyContent="space-between"
                  backgroundColor={isActive ? 'rgba(124, 82, 196, 0.08)' : 'transparent'}
                  onPress={() => {
                    loadSession(session.id);
                    setShowDrawer(false);
                  }}
                  pressStyle={{backgroundColor: 'rgba(124, 82, 196, 0.06)', scale: 0.98}}>
                  <YStack flex={1} marginRight={8}>
                    <YStack
                      onLongPress={() => {
                        triggerHaptic('light');
                        const currentTitle = session.name || 'New conversation';
                        Alert.prompt('Rename', 'Enter a new title:', (newTitle: string) => {
                          if (newTitle && newTitle.trim() && newTitle.trim() !== currentTitle) {
                            renameSession(session.id, newTitle.trim());
                          }
                        }, 'plain-text', currentTitle);
                      }}>
                      <XStack alignItems="center" gap={6}>
                        {isStarred && <Icon name="star" size={12} color={'#F59E0B'} />}
                        <Text fontSize={14} fontWeight={isActive ? '600' : '400'} color="$color" numberOfLines={1}>
                          {session.name || 'New conversation'}
                        </Text>
                      </XStack>
                    </YStack>
                    <XStack alignItems="center" gap={6} marginTop={3}>
                      <Text fontSize={11} color="$color10">
                        {session.message_count || 0} messages
                      </Text>
                      {(sessionLabels[session.id] || []).length > 0 && (
                        <XStack flexWrap="wrap" gap={3}>
                          {(sessionLabels[session.id] || []).slice(0, 2).map(label => (
                            <YStack key={label} backgroundColor="$color9" opacity={0.12} paddingHorizontal={5} paddingVertical={1} borderRadius={4}>
                              <Text fontSize={9} color="$color9">{label}</Text>
                            </YStack>
                          ))}
                        </XStack>
                      )}
                    </XStack>
                  </YStack>
                  <XStack alignItems="center" gap={2}>
                    <YStack
                      width={28} height={28} borderRadius={8}
                      alignItems="center" justifyContent="center"
                      onPress={async () => {
                        if (isStarred) {
                          await starsService.unstarSession(session.id);
                          setStarredIds(prev => prev.filter(id => id !== session.id));
                        } else {
                          await starsService.starSession(session.id);
                          setStarredIds(prev => [session.id, ...prev]);
                        }
                        triggerHaptic('light');
                      }}
                      pressStyle={{opacity: 0.6}}>
                      <Icon name={isStarred ? 'star' : 'star-outline'} size={14} color={'#F59E0B'} />
                    </YStack>
                    <YStack
                      width={28} height={28} borderRadius={8}
                      alignItems="center" justifyContent="center"
                      onPress={() => {
                        Alert.alert('Delete', 'Delete this conversation?', [
                          {text: 'Cancel', style: 'cancel'},
                          {
                            text: 'Delete',
                            style: 'destructive',
                            onPress: () => deleteSession(session.id),
                          },
                        ]);
                      }}
                      pressStyle={{opacity: 0.6}}>
                      <Icon name="trash-2" size={14} color="#EF4444" />
                    </YStack>
                  </XStack>
                </XStack>
              )}}
              ListEmptyComponent={
                <YStack padding={40} alignItems="center">
                  <Icon name="message-circle" size={24} color={(theme.color10?.val || '#9CA3AF')} />
                  <Text fontSize={13} color="$color10" marginTop={8}>No conversations yet</Text>
                </YStack>
              }
            />
            {archivedSessions.length > 0 && (
              <XStack
                paddingHorizontal={16} paddingVertical={10}
                borderTopWidth={0.5} borderTopColor="$borderColor"
                alignItems="center" justifyContent="space-between"
                onPress={() => setShowArchived(s => !s)}
                pressStyle={{backgroundColor: 'rgba(124, 82, 196, 0.04)'}}>
                <XStack alignItems="center" gap={8}>
                  <YStack width={24} height={24} borderRadius={8} backgroundColor="rgba(124, 82, 196, 0.08)" alignItems="center" justifyContent="center">
                    <Icon name="archive" size={12} color="$color9" />
                  </YStack>
                  <Text fontSize={13} fontWeight="500" color="$color">Archived</Text>
                  <YStack paddingHorizontal={6} paddingVertical={2} borderRadius={6} backgroundColor="rgba(124, 82, 196, 0.08)">
                    <Text fontSize={10} fontWeight="600" color="$color9">{archivedSessions.length}</Text>
                  </YStack>
                </XStack>
                <Text fontSize={12} color="$color10">{showArchived ? 'Hide' : 'Show'}</Text>
              </XStack>
            )}
            {showArchived && archivedSessions.length > 0 && (
              <YStack maxHeight={200}>
                <FlatList
                  data={archivedSessions}
                  keyExtractor={item => item.id}
                  renderItem={({item: session}) => (
                    <XStack
                      paddingHorizontal={16} paddingVertical={10}
                      marginHorizontal={12} marginVertical={1}
                      borderRadius={10}
                      alignItems="center" justifyContent="space-between"
                      backgroundColor={session.id === useChatStore.getState().activeSessionId ? 'rgba(124, 82, 196, 0.08)' : 'transparent'}
                      onPress={() => {
                        loadSession(session.id);
                        setShowDrawer(false);
                      }}>
                      <YStack flex={1}>
                        <Text fontSize={13} fontWeight="400" color="$color" numberOfLines={1}>
                          {session.name || 'New conversation'}
                        </Text>
                        <Text fontSize={11} color="$color10">
                          {session.message_count || 0} messages
                        </Text>
                      </YStack>
                      <YStack
                        paddingHorizontal={10} paddingVertical={4} borderRadius={8}
                        backgroundColor="rgba(124, 82, 196, 0.08)"
                        onPress={() => archiveSession(session.id, false)}>
                        <Text fontSize={11} fontWeight="500" color="$color9">Restore</Text>
                      </YStack>
                    </XStack>
                  )}
                />
              </YStack>
            )}
            {/* Label filter chips — below list */}
            {allLabels.length > 0 && (
              <XStack
                flexWrap="wrap" gap={4}
                paddingHorizontal={16} paddingVertical={8}
                paddingBottom={12}
                borderTopWidth={0.5} borderTopColor="$borderColor">
                <YStack
                  paddingHorizontal={10} paddingVertical={4} borderRadius={999}
                  backgroundColor={labelFilter === null ? '$color9' : 'transparent'}
                  borderWidth={0.5}
                  borderColor={labelFilter === null ? '$color9' : '$borderColor'}
                  onPress={() => setLabelFilter(null)}>
                  <Text fontSize={11} color={labelFilter === null ? 'white' : '$color11'}
                    fontWeight={labelFilter === null ? '600' : '400'}>All</Text>
                </YStack>
                {allLabels.map(label => (
                  <YStack
                    key={label}
                    paddingHorizontal={10} paddingVertical={4} borderRadius={999}
                    backgroundColor={labelFilter === label ? '$color9' : 'transparent'}
                    borderWidth={0.5}
                    borderColor={labelFilter === label ? '$color9' : '$borderColor'}
                    onPress={() => setLabelFilter(label === labelFilter ? null : label)}>
                    <Text fontSize={11}
                      color={labelFilter === label ? 'white' : '$color11'}
                      fontWeight={labelFilter === label ? '600' : '400'}>{label}</Text>
                  </YStack>
                ))}
              </XStack>
            )}
            {/* Safe area padding */}
            <YStack height={20} />
          </YStack>
        </YStack>
      </Modal>

      {/* Soul picker — bottom sheet */}
      <Modal visible={showSoulPicker} animationType="slide" transparent>
        <YStack flex={1} justifyContent="flex-end">
          <YStack
            flex={1}
            backgroundColor="rgba(0,0,0,0.3)"
            onPress={() => setShowSoulPicker(false)}
          />
          <YStack
            backgroundColor="$background"
            borderTopLeftRadius={24}
            borderTopRightRadius={24}
            maxHeight="75%"
            overflow="hidden">
            {/* Handle bar */}
            <YStack alignItems="center" paddingTop={12} paddingBottom={2}>
              <YStack width={40} height={5} borderRadius={3} backgroundColor="$borderColor" opacity={0.4} />
            </YStack>

            <XStack
              paddingHorizontal={20} paddingVertical={14}
              borderBottomWidth={0.5} borderBottomColor="$borderColor"
              alignItems="center" justifyContent="space-between">
              <Text fontSize={17} fontWeight="700" letterSpacing={-0.3} color="$color">Personalities</Text>
              <YStack
                width={28} height={28} borderRadius={9}
                alignItems="center" justifyContent="center"
                onPress={() => setShowSoulPicker(false)}
                pressStyle={{opacity: 0.6}}>
                <Icon name="x" size={14} color={(theme.color11?.val || '#6B7280')} />
              </YStack>
            </XStack>

            {currentSoul && (
              <YStack
                marginHorizontal={16} marginTop={12} marginBottom={4}
                paddingHorizontal={16} paddingVertical={14}
                borderRadius={14}
                backgroundColor="rgba(124, 82, 196, 0.06)"
                borderWidth={0.5} borderColor="rgba(124, 82, 196, 0.15)">
                <Text fontSize={10} fontWeight="700" letterSpacing={0.6} color="$color9" marginBottom={6}>ACTIVE</Text>
                <XStack alignItems="center" gap={10}>
                  <YStack width={32} height={32} borderRadius={16} backgroundColor="rgba(124, 82, 196, 0.12)" alignItems="center" justifyContent="center">
                    <Icon name="check" size={14} color="$color9" />
                  </YStack>
                  <YStack flex={1}>
                    <Text fontSize={15} fontWeight="600" color="$color">{currentSoul.name}</Text>
                    {currentSoul.description && (
                      <Text fontSize={12} color="$color10" marginTop={1}>{currentSoul.description}</Text>
                    )}
                  </YStack>
                </XStack>
              </YStack>
            )}

            <FlatList
              data={souls}
              keyExtractor={item => item.name}
              contentContainerStyle={{paddingVertical: 6, paddingHorizontal: 16}}
              renderItem={({item: soul}) => {
                const isActive = currentSoul?.name === soul.name;
                return (
                  <XStack
                    paddingVertical={12} paddingHorizontal={14}
                    marginVertical={2}
                    borderRadius={12}
                    alignItems="center" gap={12}
                    backgroundColor={isActive ? 'rgba(124, 82, 196, 0.08)' : 'transparent'}
                    onPress={() => {
                      switchSoul(soul.name);
                      setShowSoulPicker(false);
                    }}
                    pressStyle={{backgroundColor: 'rgba(124, 82, 196, 0.06)', scale: 0.98}}>
                    <YStack
                      width={36} height={36} borderRadius={18}
                      backgroundColor={isActive ? '$color9' : 'rgba(124, 82, 196, 0.08)'}
                      alignItems="center" justifyContent="center">
                      <Icon name="user" size={16} color={isActive ? 'white' : (theme.color9?.val || '#7C52C4')} />
                    </YStack>
                    <YStack flex={1}>
                      <Text fontSize={14} fontWeight={isActive ? '600' : '400'} color="$color">{soul.name}</Text>
                      {soul.description && (
                        <Text fontSize={11} color="$color10" numberOfLines={1} marginTop={1}>
                          {soul.description}
                        </Text>
                      )}
                      {soul.traits && soul.traits.length > 0 && (
                        <XStack flexWrap="wrap" gap={3} marginTop={4}>
                          {soul.traits.map(trait => (
                            <StatusBadge key={trait} label={trait} variant="info" />
                          ))}
                        </XStack>
                      )}
                    </YStack>
                    {isActive && (
                      <YStack width={20} height={20} borderRadius={10} backgroundColor="$color9" alignItems="center" justifyContent="center">
                        <Icon name="check" size={12} color="white" />
                      </YStack>
                    )}
                  </XStack>
                );
              }}
              ListEmptyComponent={
                <YStack padding={40} alignItems="center">
                  <Text fontSize={13} color="$color10">No personalities found</Text>
                </YStack>
              }
            />
            <YStack height={Platform.OS === 'ios' ? 34 : 16} />
          </YStack>
        </YStack>
      </Modal>

      <SearchSessionsModal
        visible={showSearchSessions}
        onClose={() => setShowSearchSessions(false)}
        onSelectSession={handleSelectSearchSession}
      />

      {/* Overflow menu — bottom sheet */}
      <Modal visible={showSettings} animationType="slide" transparent onRequestClose={() => setShowSettings(false)}>
        <YStack flex={1} justifyContent="flex-end">
          <YStack
            flex={1}
            backgroundColor="rgba(0,0,0,0.3)"
            onPress={() => setShowSettings(false)}
          />
          <YStack
            backgroundColor="$background"
            borderTopLeftRadius={24}
            borderTopRightRadius={24}
            overflow="hidden">
            {/* Handle bar */}
            <YStack alignItems="center" paddingTop={12} paddingBottom={2}>
              <YStack width={40} height={5} borderRadius={3} backgroundColor="$borderColor" opacity={0.4} />
            </YStack>

            {/* Section: Actions */}
            <XStack
              paddingHorizontal={20} paddingVertical={16}
              borderBottomWidth={0.5} borderBottomColor="$borderColor"
              alignItems="center" justifyContent="space-between">
              <Text fontSize={17} fontWeight="700" letterSpacing={-0.3} color="$color">Menu</Text>
              <YStack
                width={28} height={28} borderRadius={9}
                alignItems="center" justifyContent="center"
                onPress={() => setShowSettings(false)}
                pressStyle={{opacity: 0.6}}>
                <Icon name="x" size={14} color={(theme.color11?.val || '#6B7280')} />
              </YStack>
            </XStack>

            {/* New Chat */}
            <YStack
              paddingVertical={14} paddingHorizontal={20}
              borderBottomWidth={0.5} borderBottomColor="$borderColor"
              onPress={() => { createSession(); setShowSettings(false); }}
              pressStyle={{backgroundColor: 'rgba(124, 82, 196, 0.04)'}}>
              <XStack alignItems="center" gap={14}>
                <YStack width={36} height={36} borderRadius={12} backgroundColor="rgba(124, 82, 196, 0.08)" alignItems="center" justifyContent="center">
                  <Icon name="plus" size={18} color="$color9" />
                </YStack>
                <YStack>
                  <Text fontSize={15} fontWeight="600" color="$color">New Chat</Text>
                  <Text fontSize={12} color="$color10">Start a fresh conversation</Text>
                </YStack>
              </XStack>
            </YStack>

            {/* Search in conversation */}
            <YStack
              paddingVertical={14} paddingHorizontal={20}
              borderBottomWidth={0.5} borderBottomColor="$borderColor"
              onPress={() => { setShowSettings(false); setShowSearch(true); }}
              pressStyle={{backgroundColor: 'rgba(124, 82, 196, 0.04)'}}>
              <XStack alignItems="center" gap={14}>
                <YStack width={36} height={36} borderRadius={12} backgroundColor="rgba(124, 82, 196, 0.08)" alignItems="center" justifyContent="center">
                  <Icon name="search" size={18} color="$color9" />
                </YStack>
                <YStack>
                  <Text fontSize={15} fontWeight="600" color="$color">Search</Text>
                  <Text fontSize={12} color="$color10">Find messages in this conversation</Text>
                </YStack>
              </XStack>
            </YStack>

            {/* Theme toggle */}
            <YStack
              paddingVertical={14} paddingHorizontal={20}
              borderBottomWidth={0.5} borderBottomColor="$borderColor"
              onPress={() => {
                setShowSettings(false);
                updateTheme({theme: themeMode === 'dark' ? 'light' : 'dark'});
              }}
              pressStyle={{backgroundColor: 'rgba(124, 82, 196, 0.04)'}}>
              <XStack alignItems="center" gap={14}>
                <YStack width={36} height={36} borderRadius={12} backgroundColor="rgba(124, 82, 196, 0.08)" alignItems="center" justifyContent="center">
                  <Icon name={themeMode === 'dark' ? 'sun' : 'moon'} size={18} color="$color9" />
                </YStack>
                <YStack>
                  <Text fontSize={15} fontWeight="600" color="$color">
                    {themeMode === 'dark' ? 'Light Mode' : 'Dark Mode'}
                  </Text>
                  <Text fontSize={12} color="$color10">Currently: {themeMode === 'dark' ? 'Dark' : themeMode === 'light' ? 'Light' : 'System'}</Text>
                </YStack>
              </XStack>
            </YStack>

            {/* Conversation Summary */}
            <YStack
              paddingVertical={14} paddingHorizontal={20}
              borderBottomWidth={0.5} borderBottomColor="$borderColor"
              onPress={() => { setShowSettings(false); setShowInfo(true); }}
              pressStyle={{backgroundColor: 'rgba(124, 82, 196, 0.04)'}}>
              <XStack alignItems="center" gap={14}>
                <YStack width={36} height={36} borderRadius={12} backgroundColor="rgba(124, 82, 196, 0.08)" alignItems="center" justifyContent="center">
                  <Icon name="info" size={18} color="$color9" />
                </YStack>
                <YStack>
                  <Text fontSize={15} fontWeight="600" color="$color">Details</Text>
                  <Text fontSize={12} color="$color10">Stats, labels, and background</Text>
                </YStack>
              </XStack>
            </YStack>

            {/* Export Chat */}
            <YStack
              paddingVertical={14} paddingHorizontal={20}
              onPress={() => { setShowSettings(false); handleExportChat(); }}
              pressStyle={{backgroundColor: 'rgba(124, 82, 196, 0.04)'}}>
              <XStack alignItems="center" gap={14}>
                <YStack width={36} height={36} borderRadius={12} backgroundColor="rgba(124, 82, 196, 0.08)" alignItems="center" justifyContent="center">
                  <Icon name="download" size={18} color="$color9" />
                </YStack>
                <YStack>
                  <Text fontSize={15} fontWeight="600" color="$color">Export</Text>
                  <Text fontSize={12} color="$color10">Save conversation as markdown</Text>
                </YStack>
              </XStack>
            </YStack>

            {/* Bottom safe area */}
            <YStack height={Platform.OS === 'ios' ? 34 : 16} />
          </YStack>
        </YStack>
      </Modal>
    </SafeAreaView>
  );
}
