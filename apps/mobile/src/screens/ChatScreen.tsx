import React, {useEffect, useState, useCallback, useRef} from 'react';
import {
  FlatList,
  KeyboardAvoidingView,
  Platform,
  TouchableOpacity,
  ActivityIndicator,
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
import {pickImage, takePhoto, imageDataUrl} from '../services/image-upload';
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
        {/* Header */}
        <XStack
          paddingHorizontal={20}
          paddingVertical={14}
          borderBottomWidth={1}
          borderBottomColor="$borderColor"
          backgroundColor="$background"
          alignItems="center"
          justifyContent="space-between">
          <XStack alignItems="center" gap={8}>
            <YStack onPress={() => setShowDrawer(true)} pressStyle={{opacity: 0.6}}>
              <Icon name="menu" size={20} color={(theme.color11?.val || '#6B7280')} />
            </YStack>
            <YStack onPress={() => setShowSoulPicker(true)} pressStyle={{opacity: 0.6}}>
              <Icon name="settings" size={20} color={(theme.color11?.val || '#6B7280')} />
            </YStack>
            <YStack
              onPress={() => {
                const now = Date.now();
                if (now - lastHeaderTap.current < 300) {
                  flatListRef.current?.scrollToOffset({offset: 0, animated: true});
                }
                lastHeaderTap.current = now;
              }}>
              <Text fontSize={16} fontWeight="600" color="$color">Chat</Text>
            </YStack>
            {currentSoul && (
              <YStack
                backgroundColor="$color9"
                opacity={0.88}
                paddingHorizontal={8}
                paddingVertical={3}
                borderRadius={999}
                onPress={() => setShowSoulPicker(true)}
                pressStyle={{opacity: 0.6}}>
                <Text fontSize={11} fontWeight="600" color="white">{currentSoul.name}</Text>
              </YStack>
            )}
            <XStack
              paddingHorizontal={7}
              paddingVertical={2}
              borderRadius={999}
              backgroundColor={hybrid.activeEngine === 'slonet' ? '#22C55E33' : hybrid.activeEngine === 'qwen' ? '#F0935C33' : '#6366F133'}>
              <Text fontSize={10} fontWeight="600" letterSpacing={0.3} color="$color11">
                {hybrid.activeEngine === 'slonet'
                  ? 'SloNet'
                  : hybrid.activeEngine === 'qwen'
                  ? 'Qwen'
                  : 'Server'}
              </Text>
            </XStack>
          </XStack>

          <XStack alignItems="center" gap={8}>
            <YStack
              padding={4}
              onPress={() => {
                triggerHaptic('light');
                updateTheme({theme: isDark ? 'light' : 'dark'});
              }}
              pressStyle={{opacity: 0.6}}>
              <Icon name={isDark ? 'sun' : 'moon'} size={20} color={(theme.color11?.val || '#6B7280')} />
            </YStack>
            <YStack
              padding={4}
              onPress={() => setShowSearchSessions(true)}
              pressStyle={{opacity: 0.6}}>
              <Icon name="search" size={20} color={(theme.color11?.val || '#6B7280')} />
            </YStack>
            {messages.length > 0 && (
              <>
                <YStack
                  paddingHorizontal={8}
                  paddingVertical={5}
                  borderRadius={8}
                  backgroundColor="$background"
                  borderWidth={1}
                  borderColor="$borderColor"
                  onPress={async () => {
                    const conversationText = messages
                      .filter(m => m.content)
                      .map(m => `${m.role === 'user' ? 'User' : 'Assistant'}: ${m.content}`)
                      .join('\n');
                    const summaryPrompt = `Summarize this conversation in 3-5 bullet points:\n\n${conversationText.slice(0, 2000)}`;
                    sendMessage(summaryPrompt);
                  }}>
                  <Text fontSize={11} fontWeight="500" color="$color11">Summary</Text>
                </YStack>
                <YStack
                  paddingHorizontal={8}
                  paddingVertical={5}
                  borderRadius={8}
                  backgroundColor="$background"
                  borderWidth={1}
                  borderColor="$borderColor"
                  onPress={async () => {
                    const md = messages.map(m => {
                      const role = m.role === 'user' ? 'You' : 'Assistant';
                      return `**${role}:**\n${m.content}`;
                    }).join('\n\n---\n\n');
                    await Share.share({title: 'Chat Export', message: md});
                  }}>
                  <Text fontSize={11} fontWeight="500" color="$color11">Export</Text>
                </YStack>
              </>
            )}
            <YStack
              paddingHorizontal={12}
              paddingVertical={5}
              borderRadius={999}
              backgroundColor="$color9"
              onPress={() => createSession()}>
              <Text fontSize={11} fontWeight="600" color="white">+ New</Text>
            </YStack>
            {messages.length > 0 && (
              <YStack
                width={28} height={28} borderRadius={8}
                backgroundColor="$background"
                borderWidth={1} borderColor="$borderColor"
                alignItems="center" justifyContent="center"
                onPress={() => setShowInfo(true)}>
                <Icon name="info" size={14} color={(theme.color11?.val || '#6B7280')} />
              </YStack>
            )}
            <YStack width={8} height={8} borderRadius={999}
              backgroundColor={isConnected ? '$color9' : '$color10'} />
          </XStack>
        </XStack>

        {error && (
          <XStack
            backgroundColor="$color10"
            opacity={0.92}
            paddingHorizontal={20} paddingVertical={10}
            borderBottomWidth={1} borderBottomColor="$borderColor"
            alignItems="center" justifyContent="space-between"
            onPress={clearError}>
            <Text fontSize={12} color="$color" flex={1}>{error}</Text>
            <Icon name="x" size={18} color="#EF4444" />
          </XStack>
        )}

        {!online && (
          <XStack
            paddingHorizontal={20} paddingVertical={10}
            borderBottomWidth={1} borderBottomColor="$borderColor"
            alignItems="center" justifyContent="space-between">
            <Text fontSize={12} fontWeight="600" color="$color10">Offline</Text>
            {offlineQueue > 0 && (
              <YStack onPress={retryPendingSends}>
                <Text fontSize={11} color="$color11" textDecorationLine="underline">{offlineQueue} queued — tap to retry</Text>
              </YStack>
            )}
          </XStack>
        )}

        {showSearch && (
          <XStack
            paddingHorizontal={20} paddingVertical={8}
            backgroundColor="$background"
            borderBottomWidth={1} borderBottomColor="$borderColor"
            gap={8} alignItems="center">
            <RNTextInput
              style={{
                flex: 1, fontSize: 14, color: (theme.color?.val || '#111827'),
                backgroundColor: (theme.background?.val || '#FFFFFF'), borderRadius: 8,
                paddingHorizontal: 12, paddingVertical: 6,
                borderWidth: 1, borderColor: (theme.borderColor?.val || '#E5E7EB'),
              }}
              value={searchQuery}
              onChangeText={setSearchQuery}
              placeholder="Search messages..."
              placeholderTextColor={(theme.color10?.val || '#9CA3AF')}
              autoFocus
            />
            {searchQuery.length > 0 && (
              <YStack onPress={() => setSearchQuery('')} pressStyle={{opacity: 0.6}}>
                <Icon name="x" size={18} color={(theme.color10?.val || '#9CA3AF')} />
              </YStack>
            )}
          </XStack>
        )}

        {messages.length === 0 && !streaming ? (
          <YStack
            flex={1} alignItems="center" justifyContent="center"
            paddingHorizontal={36}
            backgroundColor={chatBackground || 'var(--background)'}>
            <YStack alignItems="center" marginBottom={16}>
              <YStack
                width={80} height={80} borderRadius={40}
                backgroundColor="$color9"
                opacity={0.12}
                alignItems="center" justifyContent="center"
                marginBottom={12}>
                <Icon name="message-circle" size={40} color={(theme.color9?.val || '#6366F1')} />
              </YStack>
              <XStack gap={6}>
                <YStack width={8} height={8} borderRadius={4} backgroundColor="$color9" />
                <YStack width={8} height={8} borderRadius={4} backgroundColor="$color9" opacity={0.7} />
                <YStack width={8} height={8} borderRadius={4} backgroundColor="$color9" opacity={0.5} />
              </XStack>
            </YStack>
            <Text fontSize={18} fontWeight="600" color="$color" marginBottom={8} textAlign="center">
              {currentSoul ? `Chat with ${currentSoul.name}` : 'Start a conversation'}
            </Text>
            <Text fontSize={14} color="$color11" textAlign="center" marginBottom={28} lineHeight={22}>
              {currentSoul
                ? `${currentSoul.description || 'Ask me anything'}`
                : 'Type a message below to begin'}
            </Text>
            <YStack gap={8} alignItems="center">
              {SUGGESTIONS.map(s => (
                <YStack
                  key={s}
                  paddingHorizontal={16} paddingVertical={10}
                  borderRadius={12}
                  borderWidth={1} borderColor="$borderColor"
                  backgroundColor="$background"
                  onPress={() => handleSuggestion(s)}
                  pressStyle={{opacity: 0.7}}>
                  <Text fontSize={12} fontWeight="600" color="$color9">{s}</Text>
                </YStack>
              ))}
            </YStack>
            <Text fontSize={11} color="$color10" marginTop={20} textAlign="center">
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
                  tintColor={(theme.color9?.val || '#6366F1')}
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
            paddingHorizontal={20} paddingVertical={8}
            backgroundColor="$color9"
            opacity={0.08}
            borderTopWidth={1} borderTopColor="$borderColor"
            gap={8} alignItems="center"
            onPress={() => {
              const firstPinnedIndex = messages.findIndex(m => pinnedIds.includes(m.id));
              if (firstPinnedIndex >= 0) {
                flatListRef.current?.scrollToIndex({index: firstPinnedIndex, animated: true, viewPosition: 0});
              }
            }}>
            <Icon name="pin" size={14} color={(theme.color9?.val || '#6366F1')} />
            <Text fontSize={11} fontWeight="600" color="$color9">
              {pinnedIds.length} pinned {pinnedIds.length === 1 ? 'message' : 'messages'} — tap to jump
            </Text>
          </XStack>
        )}

        {/* Select mode toolbar */}
        {selectMode && (
          <XStack
            paddingHorizontal={12} paddingVertical={8}
            backgroundColor="$background"
            borderTopWidth={1} borderTopColor="$borderColor"
            alignItems="center" justifyContent="space-between">
            <YStack paddingHorizontal={12} paddingVertical={4}
              onPress={() => {
                if (selectedIds.size === messages.length) {
                  setSelectedIds(new Set());
                } else {
                  setSelectedIds(new Set(messages.map(m => m.id)));
                }
              }}>
              <Text fontSize={14} fontWeight="600" color="$color9">
                {selectedIds.size === messages.length ? 'Deselect all' : 'Select all'}
              </Text>
            </YStack>
            <Text fontSize={12} color="$color10">{selectedIds.size} selected</Text>
            <YStack paddingHorizontal={12} paddingVertical={4}
              opacity={selectedIds.size === 0 ? 0.4 : 1}
              onPress={deleteSelected}
              disabled={selectedIds.size === 0}>
              <Text fontSize={14} fontWeight="600"
                color={selectedIds.size > 0 ? '$color10' : '$color9'}>Delete</Text>
            </YStack>
            <YStack paddingHorizontal={12} paddingVertical={4} onPress={toggleSelectMode}>
              <Text fontSize={14} fontWeight="600" color="$color9">Done</Text>
            </YStack>
          </XStack>
        )}

        <ReasoningPanel visible={streaming && messages.length > 0 && !messages[messages.length - 1]?.content} />

        {/* Reply quote bar */}
        {replyTo && (
          <XStack
            paddingHorizontal={12} paddingVertical={4}
            backgroundColor="$background"
            borderTopWidth={1} borderTopColor="$borderColor"
            borderLeftWidth={3} borderLeftColor="$color9"
            alignItems="center">
            <YStack flex={1} marginRight={4}>
              <Text fontSize={12} color="$color9" fontWeight="600" marginBottom={2}>
                Replying to {replyTo.role === 'user' ? 'yourself' : 'assistant'}
              </Text>
              <Text fontSize={12} color="$color10" numberOfLines={1}>
                {replyTo.content}
              </Text>
            </YStack>
            <YStack
              width={24} height={24} borderRadius={8}
              backgroundColor="$background"
              borderWidth={1} borderColor="$borderColor"
              alignItems="center" justifyContent="center"
              onPress={() => setReplyTo(null)}>
              <Icon name="x" size={12} color={(theme.color10?.val || '#9CA3AF')} />
            </YStack>
          </XStack>
        )}

        {/* Forward to session modal */}
        <Modal
          visible={forwardTo !== null}
          transparent
          animationType="slide"
          onRequestClose={() => setForwardTo(null)}>
          <YStack flex={1} justifyContent="center" padding={24}
            backgroundColor="rgba(0,0,0,0.35)"
            onPress={() => setForwardTo(null)}>
            <YStack
              backgroundColor="$background"
              borderRadius={12} maxHeight="60%" overflow="hidden"
              borderWidth={1} borderColor="$borderColor"
              onPress={(e: any) => e.stopPropagation()}>
              <Text fontSize={14} fontWeight="700" padding={12} paddingBottom={4}>
                Forward to...
              </Text>
              <Text fontSize={12} color="$color10" paddingHorizontal={12} paddingBottom={8} numberOfLines={1}>
                {forwardTo?.content}
              </Text>
              <FlatList
                data={safeSessions}
                keyExtractor={s => s.id}
                renderItem={({item: session}) => (
                  <YStack
                    paddingHorizontal={12} paddingVertical={8}
                    borderBottomWidth={1} borderBottomColor="$borderColor"
                    onPress={async () => {
                      if (forwardTo) {
                        await forwardMessage(forwardTo.content, session.id);
                      }
                      setForwardTo(null);
                    }}>
                    <Text fontSize={14} fontWeight="500" numberOfLines={1}>
                      {session.name || 'New conversation'}
                    </Text>
                    <Text fontSize={12} color="$color10">
                      {session.message_count || 0} messages
                    </Text>
                  </YStack>
                )}
              />
              <YStack padding={12} alignItems="center" borderTopWidth={1} borderTopColor="$borderColor"
                onPress={() => setForwardTo(null)}>
                <Text fontSize={14} color="$color11">Cancel</Text>
              </YStack>
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
            borderRadius={999}
            backgroundColor="$color9"
            alignItems="center"
            justifyContent="center"
            shadowColor="$color9"
            shadowOffset={{width: 0, height: 4}}
            shadowOpacity={0.3}
            shadowRadius={8}
            elevation={4}
            onPress={() => flatListRef.current?.scrollToEnd({animated: true})}>
            <Icon name="arrow-down" size={18} color="white" />
          </YStack>
        )}
      </KeyboardAvoidingView>

      {/* Conversation Info Modal */}
      <Modal visible={showInfo} transparent animationType="fade" onRequestClose={() => setShowInfo(false)}>
        <YStack flex={1} justifyContent="center" alignItems="center" padding={24}
          backgroundColor="rgba(0,0,0,0.35)"
          onPress={() => setShowInfo(false)}>
          <YStack
            backgroundColor="$background"
            borderRadius={12} padding={20} width="100%" maxWidth={300}
            borderWidth={1} borderColor="$borderColor">
            <Text fontSize={18} fontWeight="600" color="$color" marginBottom={12} textAlign="center">
              Conversation Info
            </Text>
            <XStack justifyContent="space-between" paddingVertical={8}
              borderBottomWidth={1} borderBottomColor="$borderColor">
              <Text fontSize={14} color="$color11">Messages</Text>
              <Text fontSize={14} color="$color" fontWeight="500">{messages.length}</Text>
            </XStack>
            <XStack justifyContent="space-between" paddingVertical={8}
              borderBottomWidth={1} borderBottomColor="$borderColor">
              <Text fontSize={14} color="$color11">Words</Text>
              <Text fontSize={14} color="$color" fontWeight="500">
                {messages.reduce((sum, m) => sum + (m.content?.split(/\s+/).length || 0), 0)}
              </Text>
            </XStack>
            <XStack justifyContent="space-between" paddingVertical={8}
              borderBottomWidth={1} borderBottomColor="$borderColor">
              <Text fontSize={14} color="$color11">Characters</Text>
              <Text fontSize={14} color="$color" fontWeight="500">
                {messages.reduce((sum, m) => sum + (m.content?.length || 0), 0)}
              </Text>
            </XStack>
            <XStack justifyContent="space-between" paddingVertical={8}
              borderBottomWidth={1} borderBottomColor="$borderColor">
              <Text fontSize={14} color="$color11">Session</Text>
              <Text fontSize={14} color="$color" fontWeight="500" numberOfLines={1}>
                {activeSessionId?.slice(0, 12) || 'None'}
              </Text>
            </XStack>
            {currentSoul && (
              <XStack justifyContent="space-between" paddingVertical={8}
                borderBottomWidth={1} borderBottomColor="$borderColor">
                <Text fontSize={14} color="$color11">Soul</Text>
                <Text fontSize={14} color="$color" fontWeight="500">{currentSoul.name}</Text>
              </XStack>
            )}
            <XStack justifyContent="space-between" paddingVertical={8}
              borderBottomWidth={1} borderBottomColor="$borderColor">
              <Text fontSize={14} color="$color11">Status</Text>
              <Text fontSize={14} fontWeight="500"
                color={isConnected ? '$color9' : '$color10'}>
                {isConnected ? 'Connected' : 'Offline'}
              </Text>
            </XStack>
            <YStack paddingTop={12} borderTopWidth={1} borderTopColor="$borderColor">
              <Text fontSize={12} color="$color11" fontWeight="600" marginBottom={8}>Chat Background</Text>
              <XStack flexWrap="wrap" gap={8}>
                {BG_PRESETS.map(p => {
                  const active = chatBackground === p.value;
                  return (
                    <YStack
                      key={p.value || 'none'}
                      width={36} height={36} borderRadius={8}
                      justifyContent="center" alignItems="center"
                      borderWidth={2}
                      borderColor={active ? '$color9' : 'transparent'}
                      backgroundColor={p.value ? p.value : (theme.background?.val || '#FFFFFF')}
                      style={!p.value ? {borderColor: (theme.borderColor?.val || '#E5E7EB'), backgroundColor: (theme.background?.val || '#FFFFFF')} : {}}
                      accessible
                      accessibilityLabel={p.label}
                      onPress={() => updateTheme({chatBackground: p.value})}>
                      {active && (
                        <Text fontSize={14} fontWeight="700" color="white">✓</Text>
                      )}
                    </YStack>
                  );
                })}
              </XStack>
            </YStack>
            <YStack paddingTop={12} borderTopWidth={1} borderTopColor="$borderColor">
              <Text fontSize={12} color="$color11" fontWeight="600" marginBottom={8}>Labels</Text>
              {activeSessionId && (sessionLabels[activeSessionId] || []).length > 0 && (
                <XStack flexWrap="wrap" gap={4} marginBottom={8}>
                  {(sessionLabels[activeSessionId] || []).map(label => (
                    <YStack
                      key={label}
                      backgroundColor="$color9"
                      opacity={0.12}
                      paddingHorizontal={8} paddingVertical={3} borderRadius={8}
                      onPress={async () => {
                        await labelsService.removeLabel(activeSessionId, label);
                        const labels = await labelsService.getLabels(activeSessionId);
                        setSessionLabels(prev => ({...prev, [activeSessionId]: labels}));
                        const distinct = await labelsService.getAllDistinctLabels();
                        setAllLabels(distinct);
                      }}>
                      <Text fontSize={11} color="$color9">{label} ×</Text>
                    </YStack>
                  ))}
                </XStack>
              )}
              <XStack gap={8}>
                <RNTextInput
                  style={{
                    flex: 1, fontSize: 12, color: (theme.color?.val || '#111827'),
                    backgroundColor: (theme.background?.val || '#FFFFFF'), borderRadius: 8,
                    paddingHorizontal: 8, paddingVertical: 5,
                    borderWidth: 1, borderColor: (theme.borderColor?.val || '#E5E7EB'),
                  }}
                  value={labelInput}
                  onChangeText={setLabelInput}
                  placeholder="Add label..."
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

      {/* Session drawer */}
      <Modal visible={showDrawer} animationType="slide" transparent>
        <XStack flex={1} justifyContent="flex-start">
          <YStack
            width="80%" maxWidth={320} maxHeight="80%"
            backgroundColor="$background"
            borderRightWidth={1} borderRightColor="$borderColor">
            <XStack
              paddingHorizontal={20} paddingVertical={12}
              borderBottomWidth={1} borderBottomColor="$borderColor"
              alignItems="center" justifyContent="space-between">
              <Text fontSize={16} fontWeight="600" color="$color">Conversations</Text>
              <XStack alignItems="center" gap={12}>
                <YStack onPress={() => { createSession(); setShowDrawer(false); }} pressStyle={{opacity: 0.6}}>
                  <Icon name="plus" size={20} color={(theme.color9?.val || '#6366F1')} />
                </YStack>
                <YStack onPress={() => setShowDrawer(false)} pressStyle={{opacity: 0.6}}>
                  <Icon name="x" size={20} color={(theme.color11?.val || '#6B7280')} />
                </YStack>
              </XStack>
            </XStack>

            {/* Label filter chips */}
            {allLabels.length > 0 && (
              <XStack
                flexWrap="wrap" gap={4}
                paddingHorizontal={20} paddingVertical={8}
                borderBottomWidth={1} borderBottomColor="$borderColor">
                <YStack
                  paddingHorizontal={8} paddingVertical={3} borderRadius={8}
                  backgroundColor={labelFilter === null ? '$color9' : '$background'}
                  opacity={labelFilter === null ? 1 : 0.08}
                  borderWidth={1}
                  borderColor={labelFilter === null ? '$color9' : '$borderColor'}
                  onPress={() => setLabelFilter(null)}>
                  <Text fontSize={11} color={labelFilter === null ? 'white' : '$color11'}
                    fontWeight={labelFilter === null ? '600' : '400'}>All</Text>
                </YStack>
                {allLabels.map(label => (
                  <YStack
                    key={label}
                    paddingHorizontal={8} paddingVertical={3} borderRadius={8}
                    backgroundColor={labelFilter === label ? '$color9' : '$background'}
                    opacity={labelFilter === label ? 1 : 0.08}
                    borderWidth={1}
                    borderColor={labelFilter === label ? '$color9' : '$borderColor'}
                    onPress={() => setLabelFilter(label === labelFilter ? null : label)}>
                    <Text fontSize={11}
                      color={labelFilter === label ? 'white' : '$color11'}
                      fontWeight={labelFilter === label ? '600' : '400'}>{label}</Text>
                  </YStack>
                ))}
              </XStack>
            )}

            {/* Filtered sessions */}
            <FlatList
              data={labelFilter ? sortedActiveSessions.filter(s => (sessionLabels[s.id] || []).includes(labelFilter!)) : sortedActiveSessions}
              keyExtractor={item => item.id}
              renderItem={({item: session}) => {
                const isStarred = starredIds.includes(session.id);
                return (
                <XStack
                  paddingHorizontal={20} paddingVertical={12}
                  borderBottomWidth={1} borderBottomColor="$borderColor"
                  alignItems="center" justifyContent="space-between"
                  backgroundColor={session.id === useChatStore.getState().activeSessionId ? '$color9' : '$background'}
                  opacity={session.id === useChatStore.getState().activeSessionId ? 0.08 : 1}
                  onPress={() => {
                    loadSession(session.id);
                    setShowDrawer(false);
                  }}>
                  <YStack flex={1}>
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
                      <XStack alignItems="center" gap={4}>
                        {isStarred && <Icon name="star" size={14} color={'#F59E0B'} />}
                        <Text fontSize={14} fontWeight="500" color="$color" numberOfLines={1}>
                          {session.name || 'New conversation'}
                        </Text>
                      </XStack>
                    </YStack>
                    <Text fontSize={12} color="$color10">
                      {session.message_count || 0} messages
                    </Text>
                    {(sessionLabels[session.id] || []).length > 0 && (
                      <XStack flexWrap="wrap" gap={3} marginTop={3}>
                        {(sessionLabels[session.id] || []).map(label => (
                          <YStack key={label} backgroundColor="$color9" opacity={0.12} paddingHorizontal={6} paddingVertical={2} borderRadius={8}>
                            <Text fontSize={10} color="$color9">{label}</Text>
                          </YStack>
                        ))}
                      </XStack>
                    )}
                  </YStack>
                  <XStack alignItems="center" gap={4}>
                    <YStack
                      onPress={async () => {
                        if (isStarred) {
                          await starsService.unstarSession(session.id);
                          setStarredIds(prev => prev.filter(id => id !== session.id));
                        } else {
                          await starsService.starSession(session.id);
                          setStarredIds(prev => [session.id, ...prev]);
                        }
                        triggerHaptic('light');
                      }}>
                      <Icon name={isStarred ? 'star' : 'star-outline'} size={18} color={'#F59E0B'} />
                    </YStack>
                    <YStack
                      onPress={() => {
                        Alert.alert('Archive', 'Archive this conversation?', [
                          {text: 'Cancel', style: 'cancel'},
                          {
                            text: 'Archive',
                            onPress: () => archiveSession(session.id, true),
                          },
                        ]);
                      }}>
                      <Icon name="archive" size={18} color={(theme.color10?.val || '#9CA3AF')} />
                    </YStack>
                    <YStack
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
                      <Icon name="x" size={18} color="#EF4444" />
                    </YStack>
                  </XStack>
                </XStack>
              )}}
              ListEmptyComponent={
                <Text fontSize={11} color="$color10" textAlign="center" padding={36}>
                  No conversations yet
                </Text>
              }
              ListFooterComponent={
                archivedSessions.length > 0 ? (
                  <YStack
                    paddingVertical={12} paddingHorizontal={20} alignItems="center"
                    borderTopWidth={1} borderTopColor="$borderColor"
                    onPress={() => setShowArchived(s => !s)}>
                    <Text fontSize={12} color="$color9" fontWeight="600">
                      {showArchived ? 'Hide' : 'Show'} archived ({archivedSessions.length})
                    </Text>
                  </YStack>
                ) : null
              }
            />
            {showArchived && archivedSessions.length > 0 && (
              <YStack maxHeight={200}>
                <Text fontSize={12} color="$color10" fontWeight="600"
                  paddingHorizontal={20} paddingVertical={8}
                  backgroundColor="$background" opacity={0.06}
                  borderTopWidth={1} borderTopColor="$borderColor">
                  Archived
                </Text>
                <FlatList
                  data={archivedSessions}
                  keyExtractor={item => item.id}
                  renderItem={({item: session}) => (
                    <XStack
                      paddingHorizontal={20} paddingVertical={12}
                      borderBottomWidth={1} borderBottomColor="$borderColor"
                      alignItems="center" justifyContent="space-between"
                      opacity={session.id === useChatStore.getState().activeSessionId ? 0.08 : 0.8}
                      backgroundColor={session.id === useChatStore.getState().activeSessionId ? '$color9' : '$background'}
                      onPress={() => {
                        loadSession(session.id);
                        setShowDrawer(false);
                      }}>
                      <YStack flex={1}>
                        <Text fontSize={14} fontWeight="500" color="$color" numberOfLines={1}>
                          {session.name || 'New conversation'}
                        </Text>
                        <Text fontSize={12} color="$color10">
                          {session.message_count || 0} messages
                        </Text>
                      </YStack>
                      <YStack onPress={() => archiveSession(session.id, false)}>
                        <Text fontSize={12} color="$color9" fontWeight="500">Unarchive</Text>
                      </YStack>
                    </XStack>
                  )}
                />
              </YStack>
            )}
          </YStack>
        </XStack>
      </Modal>

      {/* Soul picker */}
      <Modal visible={showSoulPicker} animationType="slide" transparent>
        <XStack flex={1} justifyContent="flex-start">
          <YStack
            width="80%" maxWidth={320} maxHeight="80%"
            backgroundColor="$background"
            borderRightWidth={1} borderRightColor="$borderColor">
            <XStack
              paddingHorizontal={20} paddingVertical={12}
              borderBottomWidth={1} borderBottomColor="$borderColor"
              alignItems="center" justifyContent="space-between">
              <Text fontSize={16} fontWeight="600" color="$color">Personality</Text>
              <YStack onPress={() => setShowSoulPicker(false)} pressStyle={{opacity: 0.6}}>
                <Icon name="x" size={20} color={(theme.color11?.val || '#6B7280')} />
              </YStack>
            </XStack>
            {currentSoul && (
              <YStack
                paddingHorizontal={20} paddingVertical={12}
                borderBottomWidth={1} borderBottomColor="$borderColor"
                backgroundColor="$color9"
                opacity={0.06}>
                <Text fontSize={12} color="$color9" fontWeight="600" marginBottom={2}>Active</Text>
                <Text fontSize={16} fontWeight="600" color="$color">{currentSoul.name}</Text>
                {currentSoul.description && (
                  <Text fontSize={11} color="$color10" marginTop={2}>{currentSoul.description}</Text>
                )}
              </YStack>
            )}
            <FlatList
              data={souls}
              keyExtractor={item => item.name}
              renderItem={({item: soul}) => {
                const isActive = currentSoul?.name === soul.name;
                return (
                  <XStack
                    paddingHorizontal={20} paddingVertical={12}
                    borderBottomWidth={1} borderBottomColor="$borderColor"
                    alignItems="center" justifyContent="space-between"
                    backgroundColor={isActive ? '$color9' : '$background'}
                    opacity={isActive ? 0.08 : 1}
                    onPress={() => {
                      switchSoul(soul.name);
                      setShowSoulPicker(false);
                    }}>
                    <YStack flex={1}>
                      <Text fontSize={14} fontWeight="500" color="$color">{soul.name}</Text>
                      {soul.description && (
                        <Text fontSize={12} color="$color10" numberOfLines={1}>
                          {soul.description}
                        </Text>
                      )}
                      {soul.traits && soul.traits.length > 0 && (
                        <XStack flexWrap="wrap" gap={4} marginTop={4}>
                          {soul.traits.map(trait => (
                            <StatusBadge key={trait} label={trait} variant="info" />
                          ))}
                        </XStack>
                      )}
                    </YStack>
                    {isActive && <Icon name="check" size={16} color={'#22C55E'} />}
                  </XStack>
                );
              }}
              ListEmptyComponent={
                <Text fontSize={11} color="$color10" textAlign="center" padding={36}>No personalities found</Text>
              }
            />
          </YStack>
        </XStack>
      </Modal>

      <SearchSessionsModal
        visible={showSearchSessions}
        onClose={() => setShowSearchSessions(false)}
        onSelectSession={handleSelectSearchSession}
      />
    </SafeAreaView>
  );
}
