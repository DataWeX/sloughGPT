import React from 'react';
import {render, fireEvent} from '@testing-library/react-native';
import {TamaguiProvider} from 'tamagui';
import appConfig from '../../../tamagui.config';
import {ChatDrawer} from '../ChatDrawer';
import {useChatStore} from '../../stores/chat-store';
import * as starsService from '../../services/stars';
import {Alert} from 'react-native';

jest.mock('../../stores/chat-store');
jest.mock('../../services/stars');
jest.mock('../../services/labels', () => ({
  addLabel: jest.fn(async () => {}),
  removeLabel: jest.fn(async () => {}),
  getLabels: jest.fn(async () => ['work']),
  getAllDistinctLabels: jest.fn(async () => ['work', 'personal']),
}));
jest.mock('../../services/haptics', () => ({
  triggerHaptic: jest.fn(),
}));
jest.mock('../../services/toast', () => ({
  toast: {error: jest.fn(), success: jest.fn(), info: jest.fn(), warn: jest.fn()},
}));
jest.mock('../Icon', () => {
  const {Text} = require('react-native');
  return {
    Icon: ({name}: any) => <Text testID={`icon-${name}`}>{name}</Text>,
  };
});

const Wrapper = ({children}: any) => (
  <TamaguiProvider config={appConfig} defaultTheme="light">{children}</TamaguiProvider>
);

const mockSessions = [
  {id: '1', name: 'First chat', created_at: '2026-01-01', updated_at: '2026-01-02', message_count: 5},
  {id: '2', name: 'Second chat', created_at: '2026-01-03', updated_at: '2026-01-04', message_count: 10},
];

const mockArchivedSessions = [
  {id: '3', name: 'Old chat', created_at: '2025-12-01', updated_at: '2025-12-02', message_count: 3, archived: true},
];

const defaultProps = {
  visible: true,
  onClose: jest.fn(),
  sortedActiveSessions: mockSessions,
  archivedSessions: mockArchivedSessions,
  sessionLabels: {1: ['work'], 2: []},
  allLabels: ['work', 'personal'],
  labelFilter: null as string | null,
  setLabelFilter: jest.fn(),
  showArchived: false,
  setShowArchived: jest.fn(),
  labelInput: '',
  setLabelInput: jest.fn(),
  starredIds: ['1'],
  setStarredIds: jest.fn(),
  setSessionLabels: jest.fn(),
  setAllLabels: jest.fn(),
};

beforeEach(() => {
  jest.clearAllMocks();
  (useChatStore.getState as jest.Mock).mockReturnValue({activeSessionId: '1'});
  (useChatStore as unknown as jest.Mock).mockReturnValue({
    loadSession: jest.fn(),
    deleteSession: jest.fn(),
    renameSession: jest.fn(),
    archiveSession: jest.fn(),
  });
  (starsService.starSession as jest.Mock).mockResolvedValue(undefined);
  (starsService.unstarSession as jest.Mock).mockResolvedValue(undefined);
});

async function renderDrawer(overrides: Partial<typeof defaultProps> = {}) {
  return render(<ChatDrawer {...defaultProps} {...overrides} />, {wrapper: Wrapper});
}

describe('ChatDrawer', () => {
  it('renders conversation list', async () => {
    const {getByText} = await renderDrawer();
    expect(getByText('Conversations')).toBeTruthy();
    expect(getByText('First chat')).toBeTruthy();
    expect(getByText('Second chat')).toBeTruthy();
  });

  it('renders message counts', async () => {
    const {getByText} = await renderDrawer();
    expect(getByText('5 messages')).toBeTruthy();
    expect(getByText('10 messages')).toBeTruthy();
  });

  it('renders label filter chips', async () => {
    const {getByText, getAllByText} = await renderDrawer();
    expect(getByText('All')).toBeTruthy();
    expect(getAllByText('work').length).toBeGreaterThanOrEqual(1);
    expect(getByText('personal')).toBeTruthy();
  });

  it('calls loadSession and onClose when session pressed', async () => {
    const mockLoad = jest.fn();
    (useChatStore as unknown as jest.Mock).mockReturnValue({
      loadSession: mockLoad,
      deleteSession: jest.fn(),
      renameSession: jest.fn(),
      archiveSession: jest.fn(),
    });
    const {getByText} = await renderDrawer();
    fireEvent.press(getByText('First chat'));
    expect(mockLoad).toHaveBeenCalledWith('1');
    expect(defaultProps.onClose).toHaveBeenCalled();
  });

  it('shows archived section when archived sessions exist', async () => {
    const {getByText} = await renderDrawer();
    expect(getByText('Archived')).toBeTruthy();
  });

  it('toggles archived visibility', async () => {
    const {getByText} = await renderDrawer();
    fireEvent.press(getByText('Archived'));
    expect(defaultProps.setShowArchived).toHaveBeenCalledWith(true);
  });

  it('shows archived sessions when showArchived is true', async () => {
    const {getByText} = await renderDrawer({showArchived: true});
    expect(getByText('Old chat')).toBeTruthy();
    expect(getByText('Restore')).toBeTruthy();
  });

  it('calls archiveSession with false on restore', async () => {
    const mockArchive = jest.fn();
    (useChatStore as unknown as jest.Mock).mockReturnValue({
      loadSession: jest.fn(),
      deleteSession: jest.fn(),
      renameSession: jest.fn(),
      archiveSession: mockArchive,
    });
    const {getByText} = await renderDrawer({showArchived: true});
    fireEvent.press(getByText('Restore'));
    expect(mockArchive).toHaveBeenCalledWith('3', false);
  });

  it('filters sessions by label', async () => {
    const {queryByText} = await renderDrawer({labelFilter: 'work'});
    expect(queryByText('First chat')).toBeTruthy();
    expect(queryByText('Second chat')).toBeNull();
  });

  it('calls setLabelFilter when label chip pressed', async () => {
    const {getAllByText} = await renderDrawer();
    const workChips = getAllByText('work');
    fireEvent.press(workChips[workChips.length - 1]);
    expect(defaultProps.setLabelFilter).toHaveBeenCalledWith('work');
  });

  it('deselects label when same label pressed', async () => {
    const {getAllByText} = await renderDrawer({labelFilter: 'work'});
    const workChips = getAllByText('work');
    fireEvent.press(workChips[workChips.length - 1]);
    expect(defaultProps.setLabelFilter).toHaveBeenCalledWith(null);
  });

  it('shows empty state when no sessions', async () => {
    const {getByText} = await renderDrawer({sortedActiveSessions: []});
    expect(getByText('No conversations yet')).toBeTruthy();
  });

  it('shows starred icon for starred sessions', async () => {
    const {getAllByTestId} = await renderDrawer();
    const starIcons = getAllByTestId('icon-star');
    expect(starIcons.length).toBeGreaterThanOrEqual(1);
  });

  it('calls deleteSession on delete confirm', async () => {
    const mockDelete = jest.fn();
    (useChatStore as unknown as jest.Mock).mockReturnValue({
      loadSession: jest.fn(),
      deleteSession: mockDelete,
      renameSession: jest.fn(),
      archiveSession: jest.fn(),
    });
    jest.spyOn(Alert, 'alert').mockImplementation((title, message, buttons) => {
      const deleteButton = (buttons as any[])?.find(b => b.text === 'Delete');
      deleteButton?.onPress?.();
    });
    const {getAllByTestId} = await renderDrawer();
    fireEvent.press(getAllByTestId('icon-trash-2')[0]);
    expect(mockDelete).toHaveBeenCalledWith('1');
  });

  it('does not delete when cancel pressed', async () => {
    const mockDelete = jest.fn();
    (useChatStore as unknown as jest.Mock).mockReturnValue({
      loadSession: jest.fn(),
      deleteSession: mockDelete,
      renameSession: jest.fn(),
      archiveSession: jest.fn(),
    });
    jest.spyOn(Alert, 'alert').mockImplementation((title, message, buttons) => {
      const cancelButton = (buttons as any[])?.find(b => b.text === 'Cancel');
      cancelButton?.onPress?.();
    });
    const {getAllByTestId} = await renderDrawer();
    fireEvent.press(getAllByTestId('icon-trash-2')[0]);
    expect(mockDelete).not.toHaveBeenCalled();
  });
});
