import {useEffect, useState, useCallback, useRef} from 'react';
import {FlatList, Alert, Share} from 'react-native';
import {useChatStore} from '../stores/chat-store';
import {useModelStore} from '../stores/model-store';
import {useOnlineStatus} from './useOnlineStatus';
import {useHybridStore} from '../stores/hybrid-inference-store';
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
import type {Message} from '../types';

export function useChatActions(flatListRef: React.RefObject<FlatList | null>) {
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
  const themeMode = useSettingsStore(s => s.theme);
  const updateTheme = useSettingsStore(s => s.update);
  const chatBackground = useSettingsStore(s => s.chatBackground);
  const online = useOnlineStatus();
  const hybrid = useHybridStore();
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

  useEffect(() => {
    if (atBottom && messages.length > 0) {
      setTimeout(() => {
        flatListRef.current?.scrollToEnd({animated: true});
      }, 50);
    }
  }, [messages, atBottom]);

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
      const stop = recordingStopRef.current;
      if (stop) {
        const recording = await stop();
        setIsRecording(false);
        recordingStopRef.current = null;
        if (recording) {
          if (voiceMessageMode) {
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
    ({item}: {item: Message}) => {
      const {MessageBubble} = require('../components/MessageBubble');
      return (
        <MessageBubble
          message={item}
          sessionId={activeSessionId || undefined}
          highlight={searchQuery ? item.content.toLowerCase().includes(searchQuery.toLowerCase()) : false}
          onRegenerate={
            item.role === 'assistant' ? () => regenerate(item.id) : undefined
          }
          onFeedback={
            item.role === 'assistant'
              ? (positive: boolean) => recordFeedback(item.id, positive)
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
      );
    },
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

  const dismissAllModals = useCallback(() => {
    setShowDrawer(false);
    setShowSoulPicker(false);
    setShowSettings(false);
    setShowSearch(false);
    setShowInfo(false);
  }, []);

  return {
    sessions,
    activeSessionId,
    messages,
    streaming,
    error,
    health,
    currentSoul,
    souls,
    switchSoul,
    themeMode,
    updateTheme,
    chatBackground,
    online,
    hybrid,
    flatListRef,
    lastHeaderTap,
    atBottom,
    setAtBottom,
    showDrawer,
    setShowDrawer,
    showSoulPicker,
    setShowSoulPicker,
    showSettings,
    setShowSettings,
    showSearch,
    setShowSearch,
    showSearchSessions,
    setShowSearchSessions,
    showInfo,
    setShowInfo,
    searchQuery,
    setSearchQuery,
    refreshing,
    setRefreshing,
    isRecording,
    setIsRecording,
    editingMessage,
    setEditingMessage,
    selectMode,
    setSelectMode,
    selectedIds,
    setSelectedIds,
    replyTo,
    setReplyTo,
    forwardTo,
    setForwardTo,
    showArchived,
    setShowArchived,
    labelFilter,
    setLabelFilter,
    sessionLabels,
    setSessionLabels,
    allLabels,
    setAllLabels,
    labelInput,
    setLabelInput,
    starredIds,
    setStarredIds,
    pinnedIds,
    setPinnedIds,
    safeSessions,
    activeSessions,
    archivedSessions,
    recordingStopRef,
    voiceMessageMode,
    setVoiceMessageMode,
    voiceTimerRef,
    sortedActiveSessions,
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
    toggleSelectMode,
    toggleSelectMessage,
    deleteSelected,
    onPullRefresh,
    handleFile,
    handleSend,
    handleImage,
    handleVoice,
    handleSuggestion,
    handleSelectSearchSession,
    handleExportChat,
    renderItem,
    keyExtractor,
    onScroll,
    isConnected,
    dismissAllModals,
  };
}
