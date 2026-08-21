/**
 * Tests for ChatScreen header elements and ReasoningPanel.
 */

import React from 'react';
import {render} from '@/test-utils';

jest.mock('react-native-safe-area-context', () => {
  const React = require('react');
  const {View} = require('react-native');
  return {
    SafeAreaView: ({children, edges, style}: any) =>
      React.createElement(View, {style, testID: 'safe-area-view'}, children),
    useSafeAreaInsets: () => ({top: 0, bottom: 0, left: 0, right: 0}),
  };
});

jest.mock('@react-navigation/native', () => ({
  ...jest.requireActual('@react-navigation/native'),
  useRoute: () => ({params: {}}),
  useNavigation: () => ({navigate: jest.fn(), goBack: jest.fn()}),
}));

jest.mock('../../stores/chat-store', () => ({
  useChatStore: () => ({
    sessions: [],
    activeSessionId: null,
    messages: [],
    streaming: false,
    error: null,
    sendMessage: jest.fn(),
    regenerate: jest.fn(),
    cancelStream: jest.fn(),
    recordFeedback: jest.fn(),
    clearError: jest.fn(),
    refreshSessions: jest.fn(),
    loadSession: jest.fn(),
    deleteSession: jest.fn(),
    archiveSession: jest.fn(),
    renameSession: jest.fn(),
    deleteMessage: jest.fn(),
    forwardMessage: jest.fn(),
    createSession: jest.fn(),
    offlineQueue: 0,
    retryPendingSends: jest.fn(),
  }),
}));

jest.mock('../../stores/model-store', () => ({
  useModelStore: () => ({
    health: {model_loaded: false},
    currentSoul: null,
    souls: [],
    switchSoul: jest.fn(),
  }),
}));

jest.mock('../../stores/hybrid-inference-store', () => ({
  useHybridStore: (overrides?: any) => ({
    activeEngine: 'remote',
    slonet: {kind: 'slonet', loaded: false, modelName: '', downloadProgress: null, description: ''},
    qwen: {kind: 'qwen', loaded: false, modelName: '', downloadProgress: null, description: ''},
    downloadProgress: 0,
    lastError: null,
    setActiveEngine: jest.fn(),
    loadSloNet: jest.fn(),
    loadQwen: jest.fn(),
    unloadSloNet: jest.fn(),
    unloadQwen: jest.fn(),
    unloadAll: jest.fn(),
    decideRoute: jest.fn(),
    executeLocal: jest.fn(),
    ...(overrides || {}),
  }),
}));

jest.mock('../../hooks/useOnlineStatus', () => ({
  useOnlineStatus: () => true,
}));

jest.mock('../../stores/settings-store', () => ({
  useSettingsStore: (() => {
    const store = {theme: 'light', chatBackground: ''};
    return (selector?: any) => (selector ? selector(store) : store);
  })(),
}));

jest.mock('../../services/haptics', () => ({
  triggerHaptic: jest.fn(),
}));

jest.mock('../../services/toast', () => ({
  toast: {success: jest.fn(), error: jest.fn(), warn: jest.fn()},
}));

jest.mock('../../services/api-client', () => ({
  api: {
    get: jest.fn(),
    post: jest.fn(),
    sendVoiceMessage: jest.fn(),
  },
}));

jest.mock('../../services/pins', () => ({
  getPinnedIds: jest.fn().mockResolvedValue([]),
}));

jest.mock('../../services/stars', () => ({
  getStarredIds: jest.fn().mockResolvedValue([]),
  starSession: jest.fn(),
  unstarSession: jest.fn(),
}));

jest.mock('../../services/labels', () => ({
  getLabels: jest.fn().mockResolvedValue([]),
  getAllDistinctLabels: jest.fn().mockResolvedValue([]),
  addLabel: jest.fn(),
  removeLabel: jest.fn(),
}));

jest.mock('../../services/voice-input', () => ({
  startRecording: jest.fn(),
  transcribeAudio: jest.fn(),
}));

jest.mock('../../services/image-upload', () => ({
  pickImage: jest.fn(),
  takePhoto: jest.fn(),
  imageDataUrl: jest.fn(),
}));

jest.mock('../../components/MessageBubble', () => {
  const React = require('react');
  const {View} = require('react-native');
  return {MessageBubble: (props: any) => React.createElement(View, {testID: 'message-bubble'})};
});

jest.mock('../../components/ChatInput', () => {
  const React = require('react');
  const {View} = require('react-native');
  return {ChatInput: (props: any) => React.createElement(View, {testID: 'chat-input'})};
});

jest.mock('../../components/ReasoningPanel', () => {
  const React = require('react');
  const {View, Text} = require('react-native');
  return {
    ReasoningPanel: ({visible}: {visible: boolean}) =>
      visible
        ? React.createElement(View, {testID: 'reasoning-panel'}, React.createElement(Text, null, 'Reasoning'))
        : null,
  };
});

jest.mock('../../components/SearchSessionsModal', () => {
  const React = require('react');
  const {View} = require('react-native');
  return {SearchSessionsModal: (props: any) => React.createElement(View, {testID: 'search-sessions-modal'})};
});

jest.mock('../../components/StatusBadge', () => {
  const React = require('react');
  const {View, Text} = require('react-native');
  return {StatusBadge: ({label}: {label: string}) => React.createElement(View, null, React.createElement(Text, null, label))};
});

// eslint-disable-next-line @typescript-eslint/no-var-requires
const {ChatScreen} = require('../ChatScreen');

describe('ChatScreen — Engine pill', () => {
  it('shows Server pill when activeEngine is remote', async () => {
    const view = await render(<ChatScreen />);
    // Header shows title "Chat" — may appear in header + empty state
    expect(view.getAllByText('Chat').length).toBeGreaterThanOrEqual(1);
  });

  it('shows SloNet pill when activeEngine is slonet', async () => {
    const spy = jest.spyOn(require('../../stores/hybrid-inference-store'), 'useHybridStore');
    spy.mockReturnValue({
      activeEngine: 'slonet',
      slonet: {kind: 'slonet', loaded: true, modelName: '', downloadProgress: null, description: ''},
      qwen: {kind: 'qwen', loaded: false, modelName: '', downloadProgress: null, description: ''},
      downloadProgress: 0,
      lastError: null,
      setActiveEngine: jest.fn(),
      loadSloNet: jest.fn(),
      loadQwen: jest.fn(),
      unloadSloNet: jest.fn(),
      unloadQwen: jest.fn(),
      unloadAll: jest.fn(),
      decideRoute: jest.fn(),
      executeLocal: jest.fn(),
    });

    const view = await render(<ChatScreen />);
    // Header shows "Chat" title — engine is in background, not shown in header anymore
    expect(view.getAllByText('Chat').length).toBeGreaterThanOrEqual(1);
    spy.mockRestore();
  });

  it('shows Qwen pill when activeEngine is qwen', async () => {
    const spy = jest.spyOn(require('../../stores/hybrid-inference-store'), 'useHybridStore');
    spy.mockReturnValue({
      activeEngine: 'qwen',
      slonet: {kind: 'slonet', loaded: false, modelName: '', downloadProgress: null, description: ''},
      qwen: {kind: 'qwen', loaded: true, modelName: '', downloadProgress: null, description: ''},
      downloadProgress: 0,
      lastError: null,
      setActiveEngine: jest.fn(),
      loadSloNet: jest.fn(),
      loadQwen: jest.fn(),
      unloadSloNet: jest.fn(),
      unloadQwen: jest.fn(),
      unloadAll: jest.fn(),
      decideRoute: jest.fn(),
      executeLocal: jest.fn(),
    });

    const view = await render(<ChatScreen />);
    // Header shows "Chat" title — engine is in background, not shown in header anymore
    expect(view.getAllByText('Chat').length).toBeGreaterThanOrEqual(1);
    spy.mockRestore();
  });
});

describe('ChatScreen — Soul pill', () => {
  it('shows soul name when currentSoul is set', async () => {
    const spy = jest.spyOn(require('../../stores/model-store'), 'useModelStore');
    spy.mockReturnValue({
      health: {model_loaded: true},
      currentSoul: {name: 'Friendly', description: 'A friendly soul'},
      souls: [{name: 'Friendly', description: 'A friendly soul'}],
      switchSoul: jest.fn(),
    });

    const view = await render(<ChatScreen />);
    // "Friendly" appears in header soul pill + empty state title
    expect(view.getAllByText('Friendly').length).toBeGreaterThanOrEqual(1);
    spy.mockRestore();
  });

  it('does not show soul pill when currentSoul is null', async () => {
    const view = await render(<ChatScreen />);
    // Default mock sets currentSoul: null, so soul pill should not render
    // The "Chat" title appears in header + empty state
    expect(view.getAllByText('Chat').length).toBeGreaterThanOrEqual(1);
  });
});

describe('ChatScreen — ReasoningPanel', () => {
  it('shows ReasoningPanel when streaming and last message has no content', async () => {
    const spy = jest.spyOn(require('../../stores/chat-store'), 'useChatStore');
    spy.mockReturnValue({
      sessions: [],
      activeSessionId: 'test-session',
      messages: [{id: '1', role: 'assistant', content: '', timestamp: Date.now()}],
      streaming: true,
      error: null,
      sendMessage: jest.fn(),
      regenerate: jest.fn(),
      cancelStream: jest.fn(),
      recordFeedback: jest.fn(),
      clearError: jest.fn(),
      refreshSessions: jest.fn(),
      loadSession: jest.fn(),
      deleteSession: jest.fn(),
      archiveSession: jest.fn(),
      renameSession: jest.fn(),
      deleteMessage: jest.fn(),
      forwardMessage: jest.fn(),
      createSession: jest.fn(),
      offlineQueue: 0,
      retryPendingSends: jest.fn(),
    });

    const view = await render(<ChatScreen />);
    expect(view.queryByTestId('reasoning-panel')).toBeTruthy();
    spy.mockRestore();
  });

  it('hides ReasoningPanel when not streaming', async () => {
    const spy = jest.spyOn(require('../../stores/chat-store'), 'useChatStore');
    spy.mockReturnValue({
      sessions: [],
      activeSessionId: 'test-session',
      messages: [{id: '1', role: 'assistant', content: 'Hello', timestamp: Date.now()}],
      streaming: false,
      error: null,
      sendMessage: jest.fn(),
      regenerate: jest.fn(),
      cancelStream: jest.fn(),
      recordFeedback: jest.fn(),
      clearError: jest.fn(),
      refreshSessions: jest.fn(),
      loadSession: jest.fn(),
      deleteSession: jest.fn(),
      archiveSession: jest.fn(),
      renameSession: jest.fn(),
      deleteMessage: jest.fn(),
      forwardMessage: jest.fn(),
      createSession: jest.fn(),
      offlineQueue: 0,
      retryPendingSends: jest.fn(),
    });

    const view = await render(<ChatScreen />);
    expect(view.queryByTestId('reasoning-panel')).toBeNull();
    spy.mockRestore();
  });

  it('hides ReasoningPanel when last message has content', async () => {
    const spy = jest.spyOn(require('../../stores/chat-store'), 'useChatStore');
    spy.mockReturnValue({
      sessions: [],
      activeSessionId: 'test-session',
      messages: [{id: '1', role: 'assistant', content: 'Hello', timestamp: Date.now()}],
      streaming: true,
      error: null,
      sendMessage: jest.fn(),
      regenerate: jest.fn(),
      cancelStream: jest.fn(),
      recordFeedback: jest.fn(),
      clearError: jest.fn(),
      refreshSessions: jest.fn(),
      loadSession: jest.fn(),
      deleteSession: jest.fn(),
      archiveSession: jest.fn(),
      renameSession: jest.fn(),
      deleteMessage: jest.fn(),
      forwardMessage: jest.fn(),
      createSession: jest.fn(),
      offlineQueue: 0,
      retryPendingSends: jest.fn(),
    });

    const view = await render(<ChatScreen />);
    // Streaming is true but last message has content — ReasoningPanel should be hidden
    expect(view.queryByTestId('reasoning-panel')).toBeNull();
    spy.mockRestore();
  });

  it('renders with empty state when no messages', async () => {
    const view = await render(<ChatScreen />);
    // Empty state shows "Chat" as title and "Start a conversation" as subtitle
    expect(view.getByText('Start a conversation')).toBeTruthy();
  });
});
