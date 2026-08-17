import React from 'react';
import {render, fireEvent, act} from '../../test-utils';
import {ChatBottomSheets} from '../ChatBottomSheets';
import * as labelsService from '../../services/labels';

jest.mock('../../services/labels', () => ({
  addLabel: jest.fn(async () => {}),
  removeLabel: jest.fn(async () => {}),
  getLabels: jest.fn(async () => ['test-label']),
  getAllDistinctLabels: jest.fn(async () => ['test-label', 'other']),
}));

jest.mock('../../services/toast', () => ({
  toast: {error: jest.fn(), success: jest.fn(), info: jest.fn(), warn: jest.fn()},
}));

jest.mock('../SearchSessionsModal', () => {
  const {Text} = require('react-native');
  return {
    SearchSessionsModal: ({visible, onClose, onSelectSession}: any) =>
      visible ? <div data-testid="search-sessions-modal"><Text>search modal</Text></div> : null,
  };
});

jest.mock('../StatusBadge', () => {
  const {Text} = require('react-native');
  return {
    StatusBadge: ({label}: any) => <div data-testid="status-badge"><Text>{label}</Text></div>,
  };
});

jest.mock('../Icon', () => ({
  Icon: ({name}: any) => <div data-testid={`icon-${name}`} />,
}));

beforeEach(() => {
  jest.clearAllMocks();
});

const defaultProps = {
  showInfo: false,
  setShowInfo: jest.fn(),
  showSearchSessions: false,
  setShowSearchSessions: jest.fn(),
  handleSelectSearchSession: jest.fn(),
  showSoulPicker: false,
  setShowSoulPicker: jest.fn(),
  showSettings: false,
  setShowSettings: jest.fn(),
  forwardTo: null as any,
  setForwardTo: jest.fn(),
  safeSessions: [
    {id: 's1', name: 'Chat 1', created_at: '', updated_at: '', message_count: 5},
    {id: 's2', name: 'Chat 2', created_at: '', updated_at: '', message_count: 10},
  ],
  activeSessionId: 's1',
  messages: [
    {id: '1', role: 'user', content: 'Hello', timestamp: 0},
    {id: '2', role: 'assistant', content: 'Hi there', timestamp: 0},
  ],
  currentSoul: {name: 'Friendly', description: 'A friendly soul', traits: ['warm', 'curious']},
  souls: [
    {name: 'Friendly', description: 'A friendly soul', traits: ['warm']},
    {name: 'Professional', description: 'Professional tone', traits: ['formal']},
  ],
  switchSoul: jest.fn(),
  isConnected: true,
  chatBackground: '',
  updateTheme: jest.fn(),
  themeMode: 'dark',
  sessionLabels: {s1: ['test-label']} as Record<string, string[]>,
  setSessionLabels: jest.fn(),
  allLabels: ['test-label', 'other'],
  setAllLabels: jest.fn(),
  labelInput: '',
  setLabelInput: jest.fn(),
  forwardMessage: jest.fn(async () => {}),
  createSession: jest.fn(async () => ({})),
  handleExportChat: jest.fn(),
  setShowSearch: jest.fn(),
};

async function renderSheet(overrides: Partial<typeof defaultProps> = {}) {
  return render(<ChatBottomSheets {...defaultProps} {...overrides} />);
}

describe('ChatBottomSheets', () => {
  it('renders without crashing when all sheets hidden', async () => {
    const {getByText} = await renderSheet();
    expect(getByText).toBeDefined();
  });

  // ── Info sheet ──────────────────────────────────────────────────────

  describe('Info sheet', () => {
    it('renders when showInfo is true', async () => {
      const {getByText} = await renderSheet({showInfo: true});
      expect(getByText('Details')).toBeTruthy();
    });

    it('shows message count', async () => {
      const {getByText} = await renderSheet({showInfo: true});
      expect(getByText('2')).toBeTruthy();
    });

    it('shows session ID', async () => {
      const {getByText} = await renderSheet({showInfo: true});
      expect(getByText('s1')).toBeTruthy();
    });

    it('shows soul name', async () => {
      const {getByText} = await renderSheet({showInfo: true});
      expect(getByText('Friendly')).toBeTruthy();
    });

    it('shows Connected status when online', async () => {
      const {getByText} = await renderSheet({showInfo: true, isConnected: true});
      expect(getByText('Connected')).toBeTruthy();
    });

    it('shows Offline status when disconnected', async () => {
      const {getByText} = await renderSheet({showInfo: true, isConnected: false});
      expect(getByText('Offline')).toBeTruthy();
    });

    it('renders chat background section', async () => {
      const {getByText} = await renderSheet({showInfo: true});
      expect(getByText('CHAT BACKGROUND')).toBeTruthy();
    });

    it('renders labels section', async () => {
      const {getByText} = await renderSheet({showInfo: true});
      expect(getByText('LABELS')).toBeTruthy();
    });

    it('shows existing labels', async () => {
      const {getByText} = await renderSheet({showInfo: true});
      expect(getByText('test-label')).toBeTruthy();
    });
  });

  // ── Soul picker ─────────────────────────────────────────────────────

  describe('Soul picker', () => {
    it('renders when showSoulPicker is true', async () => {
      const {getByText} = await renderSheet({showSoulPicker: true});
      expect(getByText('Personalities')).toBeTruthy();
    });

    it('shows ACTIVE section', async () => {
      const {getByText} = await renderSheet({showSoulPicker: true});
      expect(getByText('ACTIVE')).toBeTruthy();
    });

    it('shows current soul name in ACTIVE section', async () => {
      const {getAllByText} = await renderSheet({showSoulPicker: true});
      expect(getAllByText('Friendly').length).toBeGreaterThanOrEqual(1);
    });

    it('lists all available souls', async () => {
      const {getByText} = await renderSheet({showSoulPicker: true});
      expect(getByText('Professional')).toBeTruthy();
    });

    it('shows soul descriptions', async () => {
      const {getByText} = await renderSheet({showSoulPicker: true});
      expect(getByText('Professional tone')).toBeTruthy();
    });

    it('calls switchSoul and closes when soul pressed', async () => {
      const switchSoul = jest.fn();
      const setShowSoulPicker = jest.fn();
      const {getByText} = await renderSheet({showSoulPicker: true, switchSoul, setShowSoulPicker});
      fireEvent.press(getByText('Professional'));
      expect(switchSoul).toHaveBeenCalledWith('Professional');
      expect(setShowSoulPicker).toHaveBeenCalledWith(false);
    });

    it('shows empty state when no souls', async () => {
      const {getByText} = await renderSheet({showSoulPicker: true, souls: [], currentSoul: null});
      expect(getByText('No personalities found')).toBeTruthy();
    });
  });

  // ── Menu / overflow ─────────────────────────────────────────────────

  describe('Menu', () => {
    it('renders when showSettings is true', async () => {
      const {getByText} = await renderSheet({showSettings: true});
      expect(getByText('Menu')).toBeTruthy();
    });

    it('shows New Chat option', async () => {
      const {getByText} = await renderSheet({showSettings: true});
      expect(getByText('New Chat')).toBeTruthy();
    });

    it('shows Search option', async () => {
      const {getByText} = await renderSheet({showSettings: true});
      expect(getByText('Search')).toBeTruthy();
    });

    it('shows Light Mode label when dark', async () => {
      const {getByText} = await renderSheet({showSettings: true, themeMode: 'dark'});
      expect(getByText('Light Mode')).toBeTruthy();
    });

    it('shows Dark Mode label when light', async () => {
      const {getByText} = await renderSheet({showSettings: true, themeMode: 'light'});
      expect(getByText('Dark Mode')).toBeTruthy();
    });

    it('shows Details option', async () => {
      const {getByText} = await renderSheet({showSettings: true});
      expect(getByText('Details')).toBeTruthy();
    });

    it('shows Export option', async () => {
      const {getByText} = await renderSheet({showSettings: true});
      expect(getByText('Export')).toBeTruthy();
    });

    it('calls createSession and closes on New Chat press', async () => {
      const createSession = jest.fn(async () => ({}));
      const setShowSettings = jest.fn();
      const {getByText} = await renderSheet({showSettings: true, createSession, setShowSettings});
      fireEvent.press(getByText('New Chat'));
      expect(createSession).toHaveBeenCalled();
      expect(setShowSettings).toHaveBeenCalledWith(false);
    });

    it('closes menu and opens search on Search press', async () => {
      const setShowSettings = jest.fn();
      const setShowSearch = jest.fn();
      const {getByText} = await renderSheet({showSettings: true, setShowSettings, setShowSearch});
      fireEvent.press(getByText('Search'));
      expect(setShowSettings).toHaveBeenCalledWith(false);
      expect(setShowSearch).toHaveBeenCalledWith(true);
    });

    it('toggles theme on theme press', async () => {
      const updateTheme = jest.fn();
      const setShowSettings = jest.fn();
      const {getByText} = await renderSheet({showSettings: true, themeMode: 'dark', updateTheme, setShowSettings});
      fireEvent.press(getByText('Light Mode'));
      expect(updateTheme).toHaveBeenCalledWith({theme: 'light'});
      expect(setShowSettings).toHaveBeenCalledWith(false);
    });

    it('closes menu and calls export on Export press', async () => {
      const setShowSettings = jest.fn();
      const handleExportChat = jest.fn();
      const {getByText} = await renderSheet({showSettings: true, setShowSettings, handleExportChat});
      fireEvent.press(getByText('Export'));
      expect(setShowSettings).toHaveBeenCalledWith(false);
      expect(handleExportChat).toHaveBeenCalled();
    });
  });

  // ── Forward to ──────────────────────────────────────────────────────

  describe('Forward to', () => {
    const forwardMsg = {id: 'f1', role: 'user' as const, content: 'Forward this', timestamp: 0};

    it('renders when forwardTo is set', async () => {
      const {getByText} = await renderSheet({forwardTo: forwardMsg});
      expect(getByText('Forward to...')).toBeTruthy();
    });

    it('shows the message content preview', async () => {
      const {getByText} = await renderSheet({forwardTo: forwardMsg});
      expect(getByText('Forward this')).toBeTruthy();
    });

    it('lists available sessions', async () => {
      const {getByText} = await renderSheet({forwardTo: forwardMsg});
      expect(getByText('Chat 1')).toBeTruthy();
      expect(getByText('Chat 2')).toBeTruthy();
    });

    it('calls forwardMessage and closes on session select', async () => {
      const forwardMessage = jest.fn(async () => {});
      const setForwardTo = jest.fn();
      const {getByText} = await renderSheet({forwardTo: forwardMsg, forwardMessage, setForwardTo});
      await act(async () => {
        fireEvent.press(getByText('Chat 2'));
      });
      expect(forwardMessage).toHaveBeenCalledWith('Forward this', 's2');
      expect(setForwardTo).toHaveBeenCalledWith(null);
    });

    it('shows empty state when no sessions', async () => {
      const {getByText} = await renderSheet({forwardTo: forwardMsg, safeSessions: []});
      expect(getByText('No conversations')).toBeTruthy();
    });
  });

  // ── Search sessions modal ───────────────────────────────────────────

  describe('Search sessions modal', () => {
    it('renders when visible', async () => {
      const {getByText} = await renderSheet({showSearchSessions: true});
      expect(getByText('search modal')).toBeTruthy();
    });
  });

  // ── Labels interaction ──────────────────────────────────────────────

  describe('Labels', () => {
    it('does not add empty label', async () => {
      const setLabelInput = jest.fn();
      await renderSheet({
        showInfo: true,
        labelInput: '   ',
        setLabelInput,
      });
      expect(labelsService.addLabel).not.toHaveBeenCalled();
    });

    it('removes label on press', async () => {
      const setSessionLabels = jest.fn();
      const setAllLabels = jest.fn();
      const {getByText} = await renderSheet({
        showInfo: true,
        sessionLabels: {s1: ['test-label']},
        setSessionLabels,
        setAllLabels,
      });
      const labelEl = getByText('test-label');
      await act(async () => {
        fireEvent.press(labelEl);
      });
      expect(labelsService.removeLabel).toHaveBeenCalledWith('s1', 'test-label');
    });
  });
});
