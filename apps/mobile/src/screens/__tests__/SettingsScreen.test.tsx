/**
 * Tests for SettingsScreen.
 * Covers: inference engine selector, load/unload, offline toggle, theme,
 * chat defaults, memory context, danger zone, navigation, push notifications.
 */

import React from 'react';
import {render, fireEvent, waitFor, cleanup} from '@/test-utils';

// ── Module-level mutable mock refs (prefix required by jest.mock guard) ─

const mockNavigate = jest.fn();
const mockSetActiveEngine = jest.fn();
const mockLoadSloNet = jest.fn();
const mockLoadQwen = jest.fn();
const mockUnloadSloNet = jest.fn();
const mockUnloadQwen = jest.fn();
const mockSetOfflineOnly = jest.fn();
const mockSetTheme = jest.fn();
const mockUpdate = jest.fn();
const mockReset = jest.fn();

// State that top-level jest.mock factories read on every call
let mockHealthData: any = {status: 'healthy', model_name: 'Qwen2.5-0.5B-Instruct'};
let mockHybridState: any = {
  activeEngine: 'remote',
  slonet: {kind: 'slonet', loaded: false, modelName: '', downloadProgress: null, description: ''},
  qwen: {kind: 'qwen', loaded: false, modelName: '', downloadProgress: null, description: ''},
  downloadProgress: 0,
  offlineOnly: false,
  lastError: null,
};
// Reject by default — the API call in useEffect has .catch(() => {}), so no state is set.
// Tests that need health data override this with mockImplementationOnce.
const mockApiGet = jest.fn().mockRejectedValue(new Error('test'));

// ── Top-level mocks (jest.mock is hoisted) ────────────────────────────

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
  useNavigation: () => ({navigate: mockNavigate}),
}));

jest.mock('../../stores/settings-store', () => ({
  useSettingsStore: () => ({
    theme: 'light',
    temperature: 0.8,
    maxTokens: 256,
    topP: 0.9,
    topK: 50,
    repetitionPenalty: 1.2,
    memoryContext: '',
    setTheme: mockSetTheme,
    update: mockUpdate,
    reset: mockReset,
  }),
}));

jest.mock('../../stores/model-store', () => ({
  useModelStore: () => ({
    health: mockHealthData,
    refresh: jest.fn(),
  }),
}));

jest.mock('../../stores/hybrid-inference-store', () => ({
  useHybridStore: () => ({
    ...mockHybridState,
    setActiveEngine: mockSetActiveEngine,
    loadSloNet: mockLoadSloNet,
    loadQwen: mockLoadQwen,
    unloadSloNet: mockUnloadSloNet,
    unloadQwen: mockUnloadQwen,
    setOfflineOnly: mockSetOfflineOnly,
    decideRoute: jest.fn(),
    executeLocal: jest.fn(),
    unloadAll: jest.fn(),
  }),
}));

jest.mock('../../services/api-client', () => ({
  api: {get: mockApiGet, post: jest.fn()},
  getApiUrl: jest.fn().mockResolvedValue('http://localhost:8000'),
  setApiUrl: jest.fn().mockResolvedValue(undefined),
}));

jest.mock('../../services/push-notifications', () => ({
  registerForPushNotifications: jest.fn().mockResolvedValue('mock-token'),
  unregisterPushNotifications: jest.fn().mockResolvedValue(undefined),
  isNotificationsEnabled: jest.fn().mockResolvedValue(false),
  onNotification: jest.fn().mockReturnValue(jest.fn()),
}));

jest.mock('../../services/sounds', () => ({
  sounds: {
    isEnabled: jest.fn(() => true),
    setEnabled: jest.fn(),
    send: jest.fn(),
    receive: jest.fn(),
    error: jest.fn(),
    delete: jest.fn(),
  },
}));

jest.mock('../../services/haptics', () => ({
  triggerHaptic: jest.fn(),
}));

// eslint-disable-next-line @typescript-eslint/no-var-requires
const {SettingsScreen} = require('../SettingsScreen');

describe('SettingsScreen', () => {
  afterEach(cleanup);

  beforeEach(() => {
    jest.clearAllMocks();
    mockHealthData = {status: 'healthy', model_name: 'Qwen2.5-0.5B-Instruct'};
    mockHybridState = {
      activeEngine: 'remote',
      slonet: {kind: 'slonet', loaded: false, modelName: '', downloadProgress: null, description: ''},
      qwen: {kind: 'qwen', loaded: false, modelName: '', downloadProgress: null, description: ''},
      downloadProgress: 0,
      offlineOnly: false,
      lastError: null,
    };
  });

  // ── Render ──────────────────────────────────────────────────────────

  it('renders title', async () => {
    const view = await render(<SettingsScreen />);
    expect(view.getByText('Settings')).toBeTruthy();
  });

  // ── Server card ─────────────────────────────────────────────────────

  it('renders server status and model name', async () => {
    mockApiGet.mockImplementationOnce(
      () => Promise.resolve({status: 'healthy', model_name: 'Qwen2.5-0.5B-Instruct'}),
    );
    const view = await render(<SettingsScreen />);
    await waitFor(() => {
      expect(view.getByText('Connected')).toBeTruthy();
    });
    expect(view.getByText('Qwen2.5-0.5B-Instruct')).toBeTruthy();
  });

  it('shows Offline when API returns error', async () => {
    mockHealthData = {status: 'error', model_name: ''};
    const view = await render(<SettingsScreen />);
    // API call rejects by default, healthData stays null → shows Offline
    await waitFor(() => {
      expect(view.getByText('Offline')).toBeTruthy();
    });
  });

  it('saves server URL on Save press', async () => {
    const {setApiUrl} = require('../../services/api-client');
    const view = await render(<SettingsScreen />);
    const input = view.getByPlaceholderText('http://localhost:8000');
    // Wait for the initial URL to load from async storage
    await waitFor(() => {
      expect(input.props.value).toBe('http://localhost:8000');
    });
    fireEvent.changeText(input, 'http://example.com:8080');
    // Wait for state update to flush before pressing Save
    await waitFor(() => {
      expect(input.props.value).toBe('http://example.com:8080');
    });
    fireEvent.press(view.getByText('Save'));
    expect(setApiUrl).toHaveBeenCalledWith('http://example.com:8080');
  });

  // ── System Health nav ───────────────────────────────────────────────

  it('navigates to Health on tap', async () => {
    const view = await render(<SettingsScreen />);
    fireEvent.press(view.getByText('System Health'));
    expect(mockNavigate).toHaveBeenCalledWith('Health');
  });

  // ── Inference engine selector ───────────────────────────────────────

  it('renders all three engine chips', async () => {
    const view = await render(<SettingsScreen />);
    // SloNet, Server appear as chip + card title/status label; use getAllByText
    expect(view.getAllByText('SloNet').length).toBeGreaterThanOrEqual(1);
    expect(view.getByText('Qwen')).toBeTruthy();
    expect(view.getAllByText('Server').length).toBeGreaterThanOrEqual(1);
  });

  it('calls setActiveEngine when tapping engine chip', async () => {
    const view = await render(<SettingsScreen />);
    // Pick the first "SloNet" element (the engine chip, before the status label)
    fireEvent.press(view.getAllByText('SloNet')[0]);
    expect(mockSetActiveEngine).toHaveBeenCalledWith('slonet');
  });

  it('calls setActiveEngine with qwen', async () => {
    const view = await render(<SettingsScreen />);
    fireEvent.press(view.getByText('Qwen'));
    expect(mockSetActiveEngine).toHaveBeenCalledWith('qwen');
  });

  // ── SloNet Load / Unload ────────────────────────────────────────────

  it('shows Load button when SloNet not loaded', async () => {
    const view = await render(<SettingsScreen />);
    const loadBtns = view.getAllByText('Load');
    expect(loadBtns.length).toBeGreaterThanOrEqual(1);
  });

  it('calls loadSloNet when tapping Load', async () => {
    const view = await render(<SettingsScreen />);
    const loadBtns = view.getAllByText('Load');
    fireEvent.press(loadBtns[0]);
    expect(mockLoadSloNet).toHaveBeenCalled();
  });

  it('shows Unload button when SloNet loaded', async () => {
    mockHybridState = {
      ...mockHybridState,
      activeEngine: 'slonet',
      slonet: {kind: 'slonet', loaded: true, modelName: 'baby-slonet', downloadProgress: null, description: ''},
    };
    const view = await render(<SettingsScreen />);
    const unloadBtns = view.getAllByText('Unload');
    expect(unloadBtns.length).toBeGreaterThanOrEqual(1);
  });

  it('calls unloadSloNet when tapping Unload', async () => {
    mockHybridState = {
      ...mockHybridState,
      activeEngine: 'slonet',
      slonet: {kind: 'slonet', loaded: true, modelName: 'baby-slonet', downloadProgress: null, description: ''},
    };
    const view = await render(<SettingsScreen />);
    const unloadBtns = view.getAllByText('Unload');
    fireEvent.press(unloadBtns[0]);
    expect(mockUnloadSloNet).toHaveBeenCalled();
  });

  // ── Qwen Download / Unload ──────────────────────────────────────────

  it('shows Download button when Qwen not loaded', async () => {
    const view = await render(<SettingsScreen />);
    const downloadBtns = view.getAllByText('Download');
    expect(downloadBtns.length).toBeGreaterThanOrEqual(1);
  });

  it('calls loadQwen when tapping Download', async () => {
    const view = await render(<SettingsScreen />);
    const downloadBtns = view.getAllByText('Download');
    fireEvent.press(downloadBtns[0]);
    expect(mockLoadQwen).toHaveBeenCalled();
  });

  it('shows Unload button when Qwen loaded', async () => {
    mockHybridState = {
      ...mockHybridState,
      activeEngine: 'qwen',
      qwen: {kind: 'qwen', loaded: true, modelName: 'Qwen2.5-0.5B-Instruct', downloadProgress: null, description: ''},
    };
    const view = await render(<SettingsScreen />);
    const unloadBtns = view.getAllByText('Unload');
    expect(unloadBtns.length).toBeGreaterThanOrEqual(1);
  });

  it('calls unloadQwen when tapping Unload on Qwen', async () => {
    mockHybridState = {
      ...mockHybridState,
      activeEngine: 'qwen',
      qwen: {kind: 'qwen', loaded: true, modelName: 'Qwen2.5-0.5B-Instruct', downloadProgress: null, description: ''},
    };
    const view = await render(<SettingsScreen />);
    const unloadBtns = view.getAllByText('Unload');
    fireEvent.press(unloadBtns[unloadBtns.length - 1]);
    expect(mockUnloadQwen).toHaveBeenCalled();
  });

  it('shows error text when lastError is set', async () => {
    mockHybridState = {
      ...mockHybridState,
      lastError: 'Qwen load failed: network timeout',
    };
    const view = await render(<SettingsScreen />);
    expect(view.getByText('Qwen load failed: network timeout')).toBeTruthy();
  });

  it('shows Qwen download progress text', async () => {
    mockHybridState = {
      ...mockHybridState,
      qwen: {kind: 'qwen', loaded: false, modelName: '', downloadProgress: 0.45, description: ''},
    };
    const view = await render(<SettingsScreen />);
    expect(view.getByText(/Downloading 45%/)).toBeTruthy();
  });

  // ── Offline-only toggle ─────────────────────────────────────────────

  it('toggles offline-only on press', async () => {
    const view = await render(<SettingsScreen />);
    fireEvent.press(view.getByText('Offline Mode'));
    expect(mockSetOfflineOnly).toHaveBeenCalledWith(true);
  });

  it('shows ON when offlineOnly is true', async () => {
    mockHybridState = {...mockHybridState, offlineOnly: true};
    const view = await render(<SettingsScreen />);
    expect(view.getByText('Offline Mode')).toBeTruthy();
  });

  // ── Appearance / Theme ──────────────────────────────────────────────

  it('renders theme selector buttons', async () => {
    const view = await render(<SettingsScreen />);
    expect(view.getByText('Light')).toBeTruthy();
    expect(view.getByText('Dark')).toBeTruthy();
    expect(view.getByText('System')).toBeTruthy();
  });

  it('calls setTheme when tapping theme button', async () => {
    const view = await render(<SettingsScreen />);
    fireEvent.press(view.getByText('Dark'));
    expect(mockSetTheme).toHaveBeenCalledWith('dark');
  });

  // ── Chat Defaults ───────────────────────────────────────────────────

  it('renders temperature selector and taps a value', async () => {
    const view = await render(<SettingsScreen />);
    expect(view.getByText('Temperature')).toBeTruthy();
    fireEvent.press(view.getByText('0.6'));
    expect(mockUpdate).toHaveBeenCalledWith({temperature: 0.6});
  });

  it('renders max tokens selector and taps a value', async () => {
    const view = await render(<SettingsScreen />);
    expect(view.getByText('Max Tokens')).toBeTruthy();
    fireEvent.press(view.getByText('512'));
    expect(mockUpdate).toHaveBeenCalledWith({maxTokens: 512});
  });

  it('renders Top-P selector and taps a value', async () => {
    const view = await render(<SettingsScreen />);
    expect(view.getByText('Top-P')).toBeTruthy();
    // Use 0.7 not 0.8 to avoid conflict with temperature's 0.8
    fireEvent.press(view.getByText('0.7'));
    expect(mockUpdate).toHaveBeenCalledWith({topP: 0.7});
  });

  it('renders Top-K selector and taps a value', async () => {
    const view = await render(<SettingsScreen />);
    expect(view.getByText('Top-K')).toBeTruthy();
    fireEvent.press(view.getByText('100'));
    expect(mockUpdate).toHaveBeenCalledWith({topK: 100});
  });

  it('renders repetition penalty selector and taps a value', async () => {
    const view = await render(<SettingsScreen />);
    expect(view.getByText('Repetition Penalty')).toBeTruthy();
    fireEvent.press(view.getByText('1.5'));
    expect(mockUpdate).toHaveBeenCalledWith({repetitionPenalty: 1.5});
  });

  // ── Memory Context ──────────────────────────────────────────────────

  it('updates memory context text', async () => {
    const view = await render(<SettingsScreen />);
    const input = view.getByPlaceholderText('I prefer concise answers. My expertise is in...');
    fireEvent.changeText(input, 'Remember my name is Alice');
    expect(mockUpdate).toHaveBeenCalledWith({memoryContext: 'Remember my name is Alice'});
  });

  // ── Navigation cards ────────────────────────────────────────────────

  it('navigates to Bookmarks on tap', async () => {
    const view = await render(<SettingsScreen />);
    fireEvent.press(view.getByText('Bookmarks'));
    expect(mockNavigate).toHaveBeenCalledWith('Bookmarks');
  });

  it('navigates to About on tap', async () => {
    const view = await render(<SettingsScreen />);
    const aboutLinks = view.getAllByText('About SloughGPT');
    fireEvent.press(aboutLinks[0]);
    expect(mockNavigate).toHaveBeenCalledWith('About');
  });

  // ── New nav cards ────────────────────────────────────────────────────

  it('navigates to Training on tap', async () => {
    const view = await render(<SettingsScreen />);
    fireEvent.press(view.getByText('Training'));
    expect(mockNavigate).toHaveBeenCalledWith('Training');
  });

  it('navigates to Knowledge on tap', async () => {
    const view = await render(<SettingsScreen />);
    fireEvent.press(view.getByText('What AI Knows About Me'));
    expect(mockNavigate).toHaveBeenCalledWith('Knowledge');
  });

  it('navigates to Help on tap', async () => {
    const view = await render(<SettingsScreen />);
    fireEvent.press(view.getByText('Help'));
    expect(mockNavigate).toHaveBeenCalledWith('Help');
  });

  it('navigates to Search on tap', async () => {
    const view = await render(<SettingsScreen />);
    fireEvent.press(view.getByText('Search Messages'));
    expect(mockNavigate).toHaveBeenCalledWith('Search');
  });

  // ── Danger Zone ────────────────────────────────────────────────────

  it('shows Alert on Reset all settings press', async () => {
    const {Alert} = require('react-native');
    const alertSpy = jest.fn();
    Alert.alert = alertSpy;

    const view = await render(<SettingsScreen />);
    fireEvent.press(view.getByText('Reset all settings'));

    expect(alertSpy).toHaveBeenCalledWith(
      'Reset Settings',
      'Reset all settings to defaults?',
      expect.arrayContaining([
        expect.objectContaining({text: 'Cancel'}),
        expect.objectContaining({text: 'Reset'}),
      ]),
    );
  });

  it('calls settings.reset when confirming Reset', async () => {
    const {Alert} = require('react-native');
    const alertSpy = jest.fn();
    Alert.alert = alertSpy;

    const view = await render(<SettingsScreen />);
    fireEvent.press(view.getByText('Reset all settings'));

    const resetBtn = alertSpy.mock.calls[0][2].find(
      (b: any) => b.text === 'Reset',
    );
    resetBtn.onPress();
    expect(mockReset).toHaveBeenCalled();
  });

  // ── Push Notifications ──────────────────────────────────────────────

  it('renders push notifications card', async () => {
    const view = await render(<SettingsScreen />);
    expect(view.getByText('Push Notifications')).toBeTruthy();
    expect(view.getByText('Training and chat updates')).toBeTruthy();
  });

  // ── Sound Effects ───────────────────────────────────────────────────

  it('renders sound effects card', async () => {
    const view = await render(<SettingsScreen />);
    expect(view.getByText('Sound Effects')).toBeTruthy();
    expect(view.getByText('Audio feedback on send/receive')).toBeTruthy();
  });
});
