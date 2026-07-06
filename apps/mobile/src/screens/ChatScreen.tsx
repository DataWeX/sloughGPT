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
  RefreshControl,
} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {useChatStore} from '../stores/chat-store';
import {useModelStore} from '../stores/model-store';
import {useOnlineStatus} from '../hooks/useOnlineStatus';
import {useHybridStore} from '../stores/hybrid-inference-store';
import {MessageBubble} from '../components/MessageBubble';
import {ChatInput} from '../components/ChatInput';
import {TypingIndicator} from '../components/TypingIndicator';
import {SearchSessionsModal} from '../components/SearchSessionsModal';
import {StatusBadge} from '../components/StatusBadge';
import {triggerHaptic} from '../services/haptics';
import {api} from '../services/api-client';
import {pickImage, takePhoto, imageDataUrl} from '../services/image-upload';
import {startRecording, transcribeAudio} from '../services/voice-input';
import {toast} from '../services/toast';
import {useTheme} from '../theme/ThemeContext';
import {useSettingsStore} from '../stores/settings-store';
import {colors, spacing, radii, typography} from '../theme';
import type {Message, Session} from '../types';

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
  const {isDark} = useTheme();
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
  const activeSessions = sessions.filter(s => !s.archived);
  const archivedSessions = sessions.filter(s => s.archived);
  const recordingStopRef = useRef<(() => Promise<{uri: string; duration: number} | null>) | null>(null);
  const [voiceMessageMode, setVoiceMessageMode] = useState(false);
  const voiceTimerRef = useRef<{start: number} | null>(null);

  useEffect(() => {
    if (activeSessionId) {
      import('../services/pins').then(m => m.getPinnedIds(activeSessionId)).then(setPinnedIds);
    } else {
      setPinnedIds([]);
    }
  }, [activeSessionId]);

  useEffect(() => {
    refreshSessions();
    import('../services/stars').then(m => m.getStarredIds()).then(setStarredIds);
    import('../services/labels').then(m => {
      m.getAllDistinctLabels().then(setAllLabels);
    });
  }, []);

  useEffect(() => {
    if (showInfo && activeSessionId) {
      import('../services/labels').then(m => {
        m.getLabels(activeSessionId).then(labels => {
          setSessionLabels(prev => ({...prev, [activeSessionId]: labels}));
        });
      });
    }
  }, [showInfo, activeSessionId]);

  useEffect(() => {
    if (showDrawer) {
      import('../services/labels').then(m => m.getLabels).then(async getLabels => {
        const all: Record<string, string[]> = {};
        for (const s of sessions) {
          all[s.id] = await getLabels(s.id);
        }
        setSessionLabels(prev => ({...prev, ...all}));
        const distinct = await (await import('../services/labels')).getAllDistinctLabels();
        setAllLabels(distinct);
      });
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
                if (result.status === 'ok' || result.status === 'created') {
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
    [regenerate, recordFeedback, searchQuery, deleteMessage, activeSessionId],
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
            <TouchableOpacity
              onPress={() => {
                const now = Date.now();
                if (now - lastHeaderTap.current < 300) {
                  flatListRef.current?.scrollToOffset({offset: 0, animated: true});
                }
                lastHeaderTap.current = now;
              }}>
              <Text style={styles.title}>Chat</Text>
            </TouchableOpacity>
            {currentSoul && (
              <TouchableOpacity
                style={styles.soulPill}
                onPress={() => setShowSoulPicker(true)}>
                <Text style={styles.soulText}>{currentSoul.name}</Text>
              </TouchableOpacity>
            )}
          </View>
          <View style={styles.headerRight}>
            <TouchableOpacity
              style={styles.menuBtn}
              onPress={() => {
                triggerHaptic('light');
                updateTheme({theme: isDark ? 'light' : 'dark'});
              }}>
              <Text style={styles.menuBtn}>{isDark ? '☀️' : '🌙'}</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.menuBtn}
              onPress={() => setShowSearchSessions(true)}>
              <Text style={styles.menuBtn}>🔍</Text>
            </TouchableOpacity>
            {messages.length > 0 && (
              <>
                <TouchableOpacity
                  style={styles.exportBtn}
                  onPress={async () => {
                    // Generate summary via AI
                    const conversationText = messages
                      .filter(m => m.content)
                      .map(m => `${m.role === 'user' ? 'User' : 'Assistant'}: ${m.content}`)
                      .join('\n');
                    const summaryPrompt = `Summarize this conversation in 3-5 bullet points:\n\n${conversationText.slice(0, 2000)}`;
                    sendMessage(summaryPrompt);
                  }}>
                  <Text style={styles.exportBtnText}>Summary</Text>
                </TouchableOpacity>
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
              </>
            )}
            <TouchableOpacity
              style={styles.newChatBtn}
              onPress={() => createSession()}>
              <Text style={styles.newChatText}>+ New</Text>
            </TouchableOpacity>
            {messages.length > 0 && (
              <TouchableOpacity
                style={styles.infoBtn}
                onPress={() => setShowInfo(true)}>
                <Text style={styles.infoBtnText}>ℹ</Text>
              </TouchableOpacity>
            )}
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

        {!online && (
          <View style={styles.offlineBanner}>
            <Text style={styles.offlineText}>Offline</Text>
            {offlineQueue > 0 && (
              <TouchableOpacity onPress={retryPendingSends}>
                <Text style={styles.offlineRetry}>{offlineQueue} queued — tap to retry</Text>
              </TouchableOpacity>
            )}
          </View>
        )}

        {showSearch && (
          <View style={styles.searchBar}>
            <TextInput
              style={styles.searchInput}
              value={searchQuery}
              onChangeText={setSearchQuery}
              placeholder="Search messages..."
              placeholderTextColor={colors.textMuted}
              autoFocus
            />
            {searchQuery.length > 0 && (
              <TouchableOpacity onPress={() => setSearchQuery('')}>
                <Text style={styles.searchClear}>×</Text>
              </TouchableOpacity>
            )}
          </View>
        )}

        {messages.length === 0 && !streaming ? (
          <View style={[styles.emptyContainer, chatBackground ? {backgroundColor: chatBackground} : null]}>
            <View style={styles.emptyIllustration}>
              <Text style={styles.emptyEmoji}>💬</Text>
              <View style={styles.emptyDotRow}>
                <View style={[styles.emptyDot, {opacity: 0.3}]} />
                <View style={[styles.emptyDot, {opacity: 0.5}]} />
                <View style={[styles.emptyDot, {opacity: 0.8}]} />
              </View>
            </View>
            <Text style={styles.emptyTitle}>
              {currentSoul ? `Chat with ${currentSoul.name}` : 'Start a conversation'}
            </Text>
            <Text style={styles.emptySubtitle}>
              {currentSoul
                ? `${currentSoul.description || 'Ask me anything'}`
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
            <Text style={styles.emptyHint}>Swipe left on a message to delete it</Text>
          </View>
        ) : (
          <Pressable style={[styles.flatListContainer, chatBackground ? {backgroundColor: chatBackground} : null]} onPress={() => Keyboard.dismiss()}>
            <FlatList
              ref={flatListRef}
              data={messages}
              renderItem={renderItem}
              keyExtractor={keyExtractor}
              contentContainerStyle={styles.messageList}
              onScroll={onScroll}
              scrollEventThrottle={16}
              refreshControl={
                <RefreshControl
                  refreshing={refreshing}
                  onRefresh={onPullRefresh}
                  tintColor={colors.primary}
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
          <TouchableOpacity
            style={styles.pinnedBanner}
            onPress={() => {
              const firstPinnedIndex = messages.findIndex(m => pinnedIds.includes(m.id));
              if (firstPinnedIndex >= 0) {
                flatListRef.current?.scrollToIndex({index: firstPinnedIndex, animated: true, viewPosition: 0});
              }
            }}>
            <Text style={styles.pinnedBannerIcon}>📌</Text>
            <Text style={styles.pinnedBannerText}>
              {pinnedIds.length} pinned {pinnedIds.length === 1 ? 'message' : 'messages'} — tap to jump
            </Text>
          </TouchableOpacity>
        )}

        {/* Select mode toolbar */}
        {selectMode && (
          <View style={styles.selectToolbar}>
            <TouchableOpacity
              style={styles.selectToolbarBtn}
              onPress={() => {
                if (selectedIds.size === messages.length) {
                  setSelectedIds(new Set());
                } else {
                  setSelectedIds(new Set(messages.map(m => m.id)));
                }
              }}>
              <Text style={styles.selectToolbarText}>
                {selectedIds.size === messages.length ? 'Deselect all' : 'Select all'}
              </Text>
            </TouchableOpacity>
            <Text style={styles.selectToolbarCount}>
              {selectedIds.size} selected
            </Text>
            <TouchableOpacity
              style={[styles.selectToolbarBtn, styles.selectToolbarDelete]}
              onPress={deleteSelected}
              disabled={selectedIds.size === 0}>
              <Text style={[styles.selectToolbarText, selectedIds.size > 0 && {color: colors.error}]}>
                Delete
              </Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.selectToolbarBtn} onPress={toggleSelectMode}>
              <Text style={styles.selectToolbarText}>Done</Text>
            </TouchableOpacity>
          </View>
        )}

        <TypingIndicator visible={streaming && messages.length > 0 && !messages[messages.length - 1]?.content} />

        {/* Reply quote bar */}
        {replyTo && (
          <View style={styles.replyBar}>
            <View style={styles.replyBarContent}>
              <Text style={styles.replyBarLabel}>
                Replying to {replyTo.role === 'user' ? 'yourself' : 'assistant'}
              </Text>
              <Text style={styles.replyBarText} numberOfLines={1}>
                {replyTo.content}
              </Text>
            </View>
            <TouchableOpacity onPress={() => setReplyTo(null)} style={styles.replyBarClose}>
              <Text style={styles.replyBarCloseText}>✕</Text>
            </TouchableOpacity>
          </View>
        )}

        {/* Forward to session modal */}
        <Modal
          visible={forwardTo !== null}
          transparent
          animationType="slide"
          onRequestClose={() => setForwardTo(null)}>
          <Pressable style={styles.overlay} onPress={() => setForwardTo(null)}>
            <Pressable style={styles.forwardModal} onPress={e => e.stopPropagation()}>
              <Text style={styles.forwardTitle}>Forward to...</Text>
              <Text style={styles.forwardPreview} numberOfLines={1}>
                {forwardTo?.content}
              </Text>
              <FlatList
                data={sessions}
                keyExtractor={s => s.id}
                renderItem={({item: session}) => (
                  <TouchableOpacity
                    style={styles.forwardSessionItem}
                    onPress={async () => {
                      if (forwardTo) {
                        await forwardMessage(forwardTo.content, session.id);
                      }
                      setForwardTo(null);
                    }}>
                    <Text style={styles.forwardSessionTitle} numberOfLines={1}>
                      {session.title || 'New conversation'}
                    </Text>
                    <Text style={styles.forwardSessionMeta}>
                      {session.message_count || 0} messages
                    </Text>
                  </TouchableOpacity>
                )}
              />
              <TouchableOpacity style={styles.forwardCancel} onPress={() => setForwardTo(null)}>
                <Text style={styles.forwardCancelText}>Cancel</Text>
              </TouchableOpacity>
            </Pressable>
          </Pressable>
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
          <TouchableOpacity
            style={styles.jumpBtn}
            onPress={() =>
              flatListRef.current?.scrollToEnd({animated: true})
            }>
            <Text style={styles.jumpText}>↓</Text>
          </TouchableOpacity>
        )}
      </KeyboardAvoidingView>

      {/* Conversation Info Modal */}
      <Modal visible={showInfo} transparent animationType="fade" onRequestClose={() => setShowInfo(false)}>
        <Pressable style={styles.infoOverlay} onPress={() => setShowInfo(false)}>
          <View style={styles.infoCard}>
            <Text style={styles.infoTitle}>Conversation Info</Text>
            <View style={styles.infoRow}>
              <Text style={styles.infoLabel}>Messages</Text>
              <Text style={styles.infoValue}>{messages.length}</Text>
            </View>
            <View style={styles.infoRow}>
              <Text style={styles.infoLabel}>Words</Text>
              <Text style={styles.infoValue}>
                {messages.reduce((sum, m) => sum + (m.content?.split(/\s+/).length || 0), 0)}
              </Text>
            </View>
            <View style={styles.infoRow}>
              <Text style={styles.infoLabel}>Characters</Text>
              <Text style={styles.infoValue}>
                {messages.reduce((sum, m) => sum + (m.content?.length || 0), 0)}
              </Text>
            </View>
            <View style={styles.infoRow}>
              <Text style={styles.infoLabel}>Session</Text>
              <Text style={styles.infoValue} numberOfLines={1}>
                {activeSessionId?.slice(0, 12) || 'None'}
              </Text>
            </View>
            {currentSoul && (
              <View style={styles.infoRow}>
                <Text style={styles.infoLabel}>Soul</Text>
                <Text style={styles.infoValue}>{currentSoul.name}</Text>
              </View>
            )}
            <View style={styles.infoRow}>
              <Text style={styles.infoLabel}>Status</Text>
              <Text style={[styles.infoValue, {color: isConnected ? colors.success : colors.error}]}>
                {isConnected ? 'Connected' : 'Offline'}
              </Text>
            </View>
            <View style={styles.bgSection}>
              <Text style={styles.bgSectionLabel}>Chat Background</Text>
              <View style={styles.bgSwatchRow}>
                {BG_PRESETS.map(p => {
                  const active = chatBackground === p.value;
                  return (
                    <TouchableOpacity
                      key={p.value || 'none'}
                      style={[
                        styles.bgSwatch,
                        p.value ? {backgroundColor: p.value} : styles.bgSwatchDefault,
                        active && styles.bgSwatchActive,
                      ]}
                      accessible
                      accessibilityLabel={p.label}
                      onPress={() => updateTheme({chatBackground: p.value})}>
                      {active && (
                        <Text style={[styles.bgSwatchCheck, p.value && !p.value.startsWith('#f') && styles.bgSwatchCheckLight]}>
                          ✓
                        </Text>
                      )}
                    </TouchableOpacity>
                  );
                })}
              </View>
            </View>
            <View style={styles.bgSection}>
              <Text style={styles.bgSectionLabel}>Labels</Text>
              {activeSessionId && (sessionLabels[activeSessionId] || []).length > 0 && (
                <View style={styles.labelChipRow}>
                  {(sessionLabels[activeSessionId] || []).map(label => (
                    <TouchableOpacity
                      key={label}
                      style={styles.labelChip}
                      onPress={async () => {
                        const {removeLabel} = await import('../services/labels');
                        await removeLabel(activeSessionId, label);
                        const labels = await (await import('../services/labels')).getLabels(activeSessionId);
                        setSessionLabels(prev => ({...prev, [activeSessionId]: labels}));
                        const distinct = await (await import('../services/labels')).getAllDistinctLabels();
                        setAllLabels(distinct);
                      }}>
                      <Text style={styles.labelChipText}>{label} ×</Text>
                    </TouchableOpacity>
                  ))}
                </View>
              )}
              <View style={styles.labelAddRow}>
                <TextInput
                  style={styles.labelInput}
                  value={labelInput}
                  onChangeText={setLabelInput}
                  placeholder="Add label..."
                  placeholderTextColor={colors.textMuted}
                  returnKeyType="done"
                  onSubmitEditing={async () => {
                    if (labelInput.trim() && activeSessionId) {
                      const {addLabel} = await import('../services/labels');
                      await addLabel(activeSessionId, labelInput.trim());
                      setLabelInput('');
                      const labels = await (await import('../services/labels')).getLabels(activeSessionId);
                      setSessionLabels(prev => ({...prev, [activeSessionId]: labels}));
                      const distinct = await (await import('../services/labels')).getAllDistinctLabels();
                      setAllLabels(distinct);
                    }
                  }}
                />
              </View>
            </View>
          </View>
        </Pressable>
      </Modal>

      {/* Session drawer */}
      <Modal visible={showDrawer} animationType="slide" transparent>
        <View style={styles.drawerOverlay}>
          <View style={styles.drawer}>
            <View style={styles.drawerHeader}>
              <Text style={styles.drawerTitle}>Conversations</Text>
              <View style={{flexDirection: 'row', alignItems: 'center', gap: 12}}>
                <TouchableOpacity onPress={() => { createSession(); setShowDrawer(false); }}>
                  <Text style={{fontSize: 20, color: colors.primary}}>+</Text>
                </TouchableOpacity>
                <TouchableOpacity onPress={() => setShowDrawer(false)}>
                  <Text style={styles.drawerClose}>×</Text>
                </TouchableOpacity>
              </View>
            </View>

            {/* Label filter chips */}
            {allLabels.length > 0 && (
              <View style={styles.labelFilterRow}>
                <TouchableOpacity
                  style={[styles.labelFilterChip, labelFilter === null && styles.labelFilterChipActive]}
                  onPress={() => setLabelFilter(null)}>
                  <Text style={[styles.labelFilterText, labelFilter === null && styles.labelFilterTextActive]}>
                    All
                  </Text>
                </TouchableOpacity>
                {allLabels.map(label => (
                  <TouchableOpacity
                    key={label}
                    style={[styles.labelFilterChip, labelFilter === label && styles.labelFilterChipActive]}
                    onPress={() => setLabelFilter(label === labelFilter ? null : label)}>
                    <Text style={[styles.labelFilterText, labelFilter === label && styles.labelFilterTextActive]}>
                      {label}
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>
            )}

            {/* Filtered sessions */}
            <FlatList
              data={labelFilter ? sortedActiveSessions.filter(s => (sessionLabels[s.id] || []).includes(labelFilter!)) : sortedActiveSessions}
              keyExtractor={item => item.id}
              renderItem={({item: session}) => {
                const isStarred = starredIds.includes(session.id);
                return (
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
                    <TouchableOpacity
                      onLongPress={() => {
                        triggerHaptic('light');
                        const currentTitle = session.title || 'New conversation';
                        Alert.prompt('Rename', 'Enter a new title:', (newTitle: string) => {
                          if (newTitle && newTitle.trim() && newTitle.trim() !== currentTitle) {
                            renameSession(session.id, newTitle.trim());
                          }
                        }, 'plain-text', currentTitle);
                      }}>
                      <View style={{flexDirection: 'row', alignItems: 'center', gap: 4}}>
                        {isStarred && <Text style={styles.starIndicator}>★</Text>}
                        <Text style={styles.sessionTitle} numberOfLines={1}>
                          {session.title || 'New conversation'}
                        </Text>
                      </View>
                    </TouchableOpacity>
                    <Text style={styles.sessionMeta}>
                      {session.message_count || 0} messages
                    </Text>
                    {(sessionLabels[session.id] || []).length > 0 && (
                      <View style={styles.drawerLabelRow}>
                        {(sessionLabels[session.id] || []).map(label => (
                          <View key={label} style={styles.drawerLabelChip}>
                            <Text style={styles.drawerLabelText}>{label}</Text>
                          </View>
                        ))}
                      </View>
                    )}
                  </View>
                  <View style={{flexDirection: 'row', alignItems: 'center', gap: 4}}>
                    <TouchableOpacity
                      onPress={async () => {
                        if (isStarred) {
                          const {unstarSession} = await import('../services/stars');
                          await unstarSession(session.id);
                          setStarredIds(prev => prev.filter(id => id !== session.id));
                        } else {
                          const {starSession} = await import('../services/stars');
                          await starSession(session.id);
                          setStarredIds(prev => [session.id, ...prev]);
                        }
                        triggerHaptic('light');
                      }}>
                      <Text style={styles.starBtn}>{isStarred ? '★' : '☆'}</Text>
                    </TouchableOpacity>
                    <TouchableOpacity
                      onPress={() => {
                        Alert.alert('Archive', 'Archive this conversation?', [
                          {text: 'Cancel', style: 'cancel'},
                          {
                            text: 'Archive',
                            onPress: () => archiveSession(session.id, true),
                          },
                        ]);
                      }}>
                      <Text style={styles.sessionArchive}>📦</Text>
                    </TouchableOpacity>
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
                  </View>
                </TouchableOpacity>
              )}}
              ListEmptyComponent={
                <Text style={styles.drawerEmpty}>
                  No conversations yet
                </Text>
              }
              ListFooterComponent={
                archivedSessions.length > 0 ? (
                  <TouchableOpacity
                    style={styles.archivedToggle}
                    onPress={() => setShowArchived(s => !s)}>
                    <Text style={styles.archivedToggleText}>
                      {showArchived ? 'Hide' : 'Show'} archived ({archivedSessions.length})
                    </Text>
                  </TouchableOpacity>
                ) : null
              }
            />
            {showArchived && archivedSessions.length > 0 && (
              <View style={{maxHeight: 200}}>
                <Text style={styles.archivedHeader}>Archived</Text>
                <FlatList
                  data={archivedSessions}
                  keyExtractor={item => item.id}
                  renderItem={({item: session}) => (
                    <TouchableOpacity
                      style={[
                        styles.sessionItem,
                        styles.archivedItem,
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
                        onPress={() => archiveSession(session.id, false)}>
                        <Text style={styles.sessionUnarchive}>Unarchive</Text>
                      </TouchableOpacity>
                    </TouchableOpacity>
                  )}
                />
              </View>
            )}
          </View>
        </View>
      </Modal>

      {/* Soul picker */}
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

      <SearchSessionsModal
        visible={showSearchSessions}
        onClose={() => setShowSearchSessions(false)}
        onSelectSession={handleSelectSearchSession}
      />
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
  infoBtn: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: colors.surface,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: colors.border,
  },
  infoBtnText: {
    fontSize: 14,
    color: colors.textSecondary,
  },
  infoOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.4)',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 24,
  },
  infoCard: {
    backgroundColor: colors.surface,
    borderRadius: radii.lg,
    padding: spacing.lg,
    width: '100%',
    maxWidth: 300,
  },
  infoTitle: {
    ...typography.h2,
    color: colors.text,
    marginBottom: spacing.md,
    textAlign: 'center',
  },
  infoRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: spacing.sm,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  infoLabel: {
    ...typography.body,
    color: colors.textSecondary,
  },
  infoValue: {
    ...typography.body,
    color: colors.text,
    fontWeight: '500',
    maxWidth: '60%',
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
  offlineBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: '#FFF3CD',
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
  },
  offlineText: {
    ...typography.caption,
    color: '#856404',
    fontWeight: '600',
  },
  offlineRetry: {
    ...typography.small,
    color: '#856404',
    textDecorationLine: 'underline',
  },
  searchBar: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
    backgroundColor: colors.surface,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
    gap: spacing.sm,
  },
  searchInput: {
    flex: 1,
    ...typography.body,
    color: colors.text,
    backgroundColor: colors.background,
    borderRadius: radii.md,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.xs,
  },
  searchClear: {
    fontSize: 18,
    color: colors.textMuted,
    padding: spacing.xs,
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
  emptyIllustration: {
    alignItems: 'center',
    marginBottom: spacing.lg,
  },
  emptyEmoji: {
    fontSize: 56,
    marginBottom: spacing.sm,
  },
  emptyDotRow: {
    flexDirection: 'row',
    gap: 6,
  },
  emptyDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: colors.primary,
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
  emptyHint: {
    ...typography.small,
    color: colors.textMuted,
    marginTop: spacing.xxl,
    textAlign: 'center',
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
  selectToolbar: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    backgroundColor: colors.surface,
    borderTopWidth: 1,
    borderTopColor: colors.border,
  },
  selectToolbarBtn: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.xs,
  },
  selectToolbarText: {
    ...typography.body,
    color: colors.primary,
    fontWeight: '600',
  },
  selectToolbarCount: {
    ...typography.small,
    color: colors.textMuted,
  },
  selectToolbarDelete: {},
  replyBar: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.xs,
    backgroundColor: colors.surface,
    borderTopWidth: 1,
    borderTopColor: colors.border,
    borderLeftWidth: 3,
    borderLeftColor: colors.primary,
  },
  replyBarContent: {
    flex: 1,
    marginRight: spacing.xs,
  },
  replyBarLabel: {
    ...typography.small,
    color: colors.primary,
    fontWeight: '600',
    marginBottom: 2,
  },
  replyBarText: {
    ...typography.small,
    color: colors.textMuted,
  },
  replyBarClose: {
    width: 24,
    height: 24,
    borderRadius: 12,
    backgroundColor: colors.surface,
    justifyContent: 'center',
    alignItems: 'center',
  },
  replyBarCloseText: {
    color: colors.textMuted,
    fontSize: 12,
  },
  forwardModal: {
    backgroundColor: colors.surface,
    borderRadius: radii.lg,
    margin: spacing.xl,
    maxHeight: '60%',
    overflow: 'hidden',
  },
  forwardTitle: {
    ...typography.body,
    fontWeight: '700',
    padding: spacing.md,
    paddingBottom: spacing.xs,
  },
  forwardPreview: {
    ...typography.small,
    color: colors.textMuted,
    paddingHorizontal: spacing.md,
    paddingBottom: spacing.sm,
  },
  forwardSessionItem: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  forwardSessionTitle: {
    ...typography.body,
    fontWeight: '500',
  },
  forwardSessionMeta: {
    ...typography.small,
    color: colors.textMuted,
  },
  forwardCancel: {
    padding: spacing.md,
    alignItems: 'center',
    borderTopWidth: 1,
    borderTopColor: colors.border,
  },
  forwardCancelText: {
    ...typography.body,
    color: colors.textMuted,
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
  activeSoul: {
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
    backgroundColor: colors.primary + '08',
  },
  activeSoulLabel: {
    ...typography.small,
    color: colors.primary,
    fontWeight: '600',
    marginBottom: 2,
  },
  activeSoulName: {
    ...typography.h3,
    color: colors.text,
  },
  activeSoulDesc: {
    ...typography.caption,
    color: colors.textMuted,
    marginTop: 2,
  },
  traitRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 4,
    marginTop: 4,
  },
  sessionArchive: {
    fontSize: 16,
    padding: spacing.xs,
  },
  sessionUnarchive: {
    fontSize: 12,
    color: colors.primary,
    fontWeight: '500',
    padding: spacing.xs,
  },
  archivedToggle: {
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.lg,
    alignItems: 'center',
    borderTopWidth: 1,
    borderTopColor: colors.border,
  },
  archivedToggleText: {
    ...typography.small,
    color: colors.primary,
    fontWeight: '600',
  },
  archivedHeader: {
    ...typography.small,
    color: colors.textMuted,
    fontWeight: '600',
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
    backgroundColor: colors.surface,
    borderTopWidth: 1,
    borderTopColor: colors.border,
  },
  archivedItem: {
    opacity: 0.8,
  },
  overlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.4)',
    justifyContent: 'center',
    padding: 24,
  },
  pinnedBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
    backgroundColor: colors.primary + '10',
    borderTopWidth: 1,
    borderTopColor: colors.primary + '30',
  },
  pinnedBannerIcon: {
    fontSize: 14,
  },
  pinnedBannerText: {
    ...typography.small,
    color: colors.primary,
    fontWeight: '600',
  },
  starBtn: {
    fontSize: 18,
    color: '#f5a623',
    padding: spacing.xs,
  },
  starIndicator: {
    fontSize: 14,
    color: '#f5a623',
  },
  flatListContainer: {
    flex: 1,
  },
  labelFilterRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.xs,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  labelFilterChip: {
    backgroundColor: colors.surface,
    paddingHorizontal: spacing.sm,
    paddingVertical: 2,
    borderRadius: radii.full,
    borderWidth: 1,
    borderColor: colors.border,
  },
  labelFilterChipActive: {
    backgroundColor: colors.primary + '20',
    borderColor: colors.primary,
  },
  labelFilterText: {
    ...typography.small,
    color: colors.textSecondary,
    fontSize: 11,
  },
  labelFilterTextActive: {
    color: colors.primary,
    fontWeight: '600',
  },
  labelChipRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.xs,
    marginBottom: spacing.sm,
  },
  labelChip: {
    backgroundColor: colors.primary + '15',
    paddingHorizontal: spacing.sm,
    paddingVertical: 2,
    borderRadius: radii.full,
  },
  labelChipText: {
    ...typography.small,
    color: colors.primary,
    fontSize: 11,
  },
  labelAddRow: {
    flexDirection: 'row',
    gap: spacing.sm,
  },
  labelInput: {
    flex: 1,
    ...typography.small,
    color: colors.text,
    backgroundColor: colors.background,
    borderRadius: radii.md,
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
    borderWidth: 1,
    borderColor: colors.border,
  },
  drawerLabelRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 3,
    marginTop: 3,
  },
  drawerLabelChip: {
    backgroundColor: colors.primary + '15',
    paddingHorizontal: 6,
    paddingVertical: 1,
    borderRadius: radii.full,
  },
  drawerLabelText: {
    fontSize: 10,
    color: colors.primary,
  },
  bgSection: {
    paddingTop: spacing.md,
    borderTopWidth: 1,
    borderTopColor: colors.border,
  },
  bgSectionLabel: {
    ...typography.small,
    color: colors.textSecondary,
    fontWeight: '600',
    marginBottom: spacing.sm,
  },
  bgSwatchRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.sm,
  },
  bgSwatch: {
    width: 36,
    height: 36,
    borderRadius: radii.full,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 2,
    borderColor: 'transparent',
  },
  bgSwatchDefault: {
    backgroundColor: colors.surface,
    borderColor: colors.border,
  },
  bgSwatchActive: {
    borderColor: colors.primary,
  },
  bgSwatchCheck: {
    fontSize: 14,
    color: '#fff',
    fontWeight: '700',
  },
  bgSwatchCheckLight: {
    color: '#fff',
  },
});
