import {useEffect, useState, useCallback, useRef} from 'react';
import {FlatList} from 'react-native';
import {useChatStore} from '../stores/chat-store';
import {useModelStore} from '../stores/model-store';
import {useOnlineStatus} from './useOnlineStatus';
import {useHybridStore} from '../stores/hybrid-inference-store';
import {triggerHaptic} from '../services/haptics';
import {api} from '../services/api-client';
import {pickImage, imageDataUrl} from '../services/image-upload';
import {toast} from '../services/toast';
import {useSettingsStore} from '../stores/settings-store';
import {shareConversation} from '../services/conversation-export';
import * as labelsService from '../services/labels';
import type {Message} from '../types';
import {useChatModals} from './chat/useChatModals';
import {useMessageSelect} from './chat/useMessageSelect';
import {useChatVoice} from './chat/useChatVoice';
import {useChatLabels} from './chat/useChatLabels';
import {useChatScroll} from './chat/useChatScroll';

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

  const [searchQuery, setSearchQuery] = useState('');
  const [refreshing, setRefreshing] = useState(false);
  const [editingMessage, setEditingMessage] = useState<string | null>(null);
  const [replyTo, setReplyTo] = useState<Message | null>(null);
  const [forwardTo, setForwardTo] = useState<Message | null>(null);

  const modals = useChatModals();
  const select = useMessageSelect(deleteMessage);
  const voice = useChatVoice(sendMessage, activeSessionId, createSession);
  const labels = useChatLabels(sessions, activeSessionId, loadSession, refreshSessions);
  const scroll = useChatScroll(
    flatListRef, messages, streaming, activeSessionId,
    regenerate, recordFeedback, deleteMessage,
    searchQuery, select.selectMode, select.selectedIds,
    select.toggleSelectMessage, setEditingMessage,
    setReplyTo, setForwardTo, select.setSelectMode, select.setSelectedIds,
  );

  useEffect(() => {
    if (modals.showInfo && activeSessionId) {
      labelsService.getLabels(activeSessionId).then(fetchedLabels => {
        labels.setSessionLabels(prev => ({...prev, [activeSessionId]: fetchedLabels}));
      });
    }
  }, [modals.showInfo, activeSessionId]);

  useEffect(() => {
    if (modals.showDrawer) {
      (async () => {
        const all: Record<string, string[]> = {};
        for (const s of (sessions ?? [])) {
          all[s.id] = await labelsService.getLabels(s.id);
        }
        labels.setSessionLabels(prev => ({...prev, ...all}));
        const distinct = await labelsService.getAllDistinctLabels();
        labels.setAllLabels(distinct);
      })();
    }
  }, [modals.showDrawer, sessions]);

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

  const handleSendWithImages = useCallback(
    (text: string, images: string[]) => {
      sendMessage(text, images);
    },
    [sendMessage],
  );

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
    const session = sessions.find(s => s.id === activeSessionId);
    const shared = await shareConversation(messages, session?.name);
    if (shared) {
      triggerHaptic('success');
      toast.success('Conversation exported');
    }
  }, [activeSessionId, messages, sessions]);

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
    lastHeaderTap: scroll.lastHeaderTap,
    atBottom: scroll.atBottom,
    setAtBottom: scroll.setAtBottom,
    showDrawer: modals.showDrawer,
    setShowDrawer: modals.setShowDrawer,
    showSoulPicker: modals.showSoulPicker,
    setShowSoulPicker: modals.setShowSoulPicker,
    showSettings: modals.showSettings,
    setShowSettings: modals.setShowSettings,
    showChatSettings: modals.showChatSettings,
    setShowChatSettings: modals.setShowChatSettings,
    showSystemPrompt: modals.showSystemPrompt,
    setShowSystemPrompt: modals.setShowSystemPrompt,
    showSearch: modals.showSearch,
    setShowSearch: modals.setShowSearch,
    showSearchSessions: modals.showSearchSessions,
    setShowSearchSessions: modals.setShowSearchSessions,
    showInfo: modals.showInfo,
    setShowInfo: modals.setShowInfo,
    searchQuery,
    setSearchQuery,
    refreshing,
    setRefreshing,
    isRecording: voice.isRecording,
    setIsRecording: voice.setIsRecording,
    editingMessage,
    setEditingMessage,
    selectMode: select.selectMode,
    setSelectMode: select.setSelectMode,
    selectedIds: select.selectedIds,
    setSelectedIds: select.setSelectedIds,
    replyTo,
    setReplyTo,
    forwardTo,
    setForwardTo,
    showArchived: labels.showArchived,
    setShowArchived: labels.setShowArchived,
    labelFilter: labels.labelFilter,
    setLabelFilter: labels.setLabelFilter,
    sessionLabels: labels.sessionLabels,
    setSessionLabels: labels.setSessionLabels,
    allLabels: labels.allLabels,
    setAllLabels: labels.setAllLabels,
    labelInput: labels.labelInput,
    setLabelInput: labels.setLabelInput,
    starredIds: labels.starredIds,
    setStarredIds: labels.setStarredIds,
    pinnedIds: labels.pinnedIds,
    setPinnedIds: labels.setPinnedIds,
    safeSessions: labels.safeSessions,
    activeSessions: labels.activeSessions,
    archivedSessions: labels.archivedSessions,
    recordingStopRef: voice.recordingStopRef,
    voiceMessageMode: voice.voiceMessageMode,
    setVoiceMessageMode: voice.setVoiceMessageMode,
    voiceTimerRef: voice.voiceTimerRef,
    sortedActiveSessions: labels.sortedActiveSessions,
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
    toggleSelectMode: select.toggleSelectMode,
    toggleSelectMessage: select.toggleSelectMessage,
    deleteSelected: select.deleteSelected,
    onPullRefresh,
    handleFile,
    handleSend,
    handleImage,
    handleSendWithImages,
    handleVoice: voice.handleVoice,
    handleSuggestion,
    handleSelectSearchSession,
    handleExportChat,
    renderItem: scroll.renderItem,
    keyExtractor: scroll.keyExtractor,
    onScroll: scroll.onScroll,
    isConnected: online,
    dismissAllModals: modals.dismissAllModals,
  };
}
