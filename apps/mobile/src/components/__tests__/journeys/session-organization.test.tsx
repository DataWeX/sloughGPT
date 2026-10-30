/**
 * Journey: Mobile: Organize My Conversations (UX_FLOWS.md)
 *
 * In-code journey test — drives the real ChatDrawer against the real
 * folders service and the real chat-store, with only the transport
 * (api-client) mocked. Mapped in domain/journeys/flows.py as m1-organize.
 */
import React from 'react';
import {Alert} from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import {act, fireEvent, render, waitFor} from '../../../test-utils';
import {ChatDrawer} from '../../ChatDrawer';
import {api} from '../../../services/api-client';
import * as foldersService from '../../../services/folders';
import {useChatStore} from '../../../stores/chat-store';

jest.mock('../../../services/api-client');
jest.mock('../../../services/sse-client');
jest.mock('../../../services/offline-cache');
jest.mock('../../../services/haptics');
jest.mock('../../../services/sounds');
jest.mock('../../../services/toast');
jest.mock('../../../stores/settings-store', () => {
  const state = {
    theme: 'dark',
    temperature: 0.8,
    maxTokens: 256,
    topP: 0.9,
    topK: 40,
    repetitionPenalty: 1.1,
  };
  const useSettingsStore: any = (selector: (s: typeof state) => unknown) => selector(state);
  useSettingsStore.getState = () => state;
  useSettingsStore.setState = () => state;
  useSettingsStore.subscribe = () => () => {};
  return {useSettingsStore, __esModule: true};
});
jest.mock('../../../stores/hybrid-inference-store', () => ({
  useHybridStore: {
    getState: () => ({
      decideRoute: () => ({target: 'remote'}),
      executeLocal: jest.fn(),
      offlineOnly: false,
    }),
  },
}));

const mockApi = api as jest.Mocked<typeof api>;

const SEED_SESSIONS = [
  {id: 's1', name: 'Trip plans', created_at: '2026-01-01', updated_at: '2026-01-02', message_count: 3},
  {id: 's2', name: 'Recipes', created_at: '2026-01-03', updated_at: '2026-01-04', message_count: 7},
];

type AlertButton = {text?: string; onPress?: () => void; style?: string};

let liveSessions: typeof SEED_SESSIONS;
let alertCalls: {title: string; buttons: AlertButton[]}[] = [];
let promptCb: ((value: string) => void) | null = null;
let cloneCounter = 0;

function wireStorage() {
  const store = new Map<string, string>();
  jest.spyOn(AsyncStorage, 'getItem').mockImplementation(async key => store.get(key) ?? null);
  jest.spyOn(AsyncStorage, 'setItem').mockImplementation(async (key, value) => {
    store.set(key, String(value));
  });
  jest.spyOn(AsyncStorage, 'removeItem').mockImplementation(async key => {
    store.delete(key);
  });
  jest.spyOn(AsyncStorage, 'multiRemove').mockImplementation(async keys => {
    keys.forEach(k => store.delete(k));
  });
}

function wireApi() {
  (mockApi.get as jest.Mock).mockImplementation(async (path: string) => {
    if (path === '/chat/sessions') {
      return [...liveSessions];
    }
    const m = /^\/session\/([^/]+)\/messages$/.exec(path);
    if (m) {
      return {
        messages: [{id: `${m[1]}-m1`, role: 'user', content: 'hello', timestamp: 0}],
      };
    }
    throw new Error(`unexpected GET ${path}`);
  });
  (mockApi.post as jest.Mock).mockImplementation(async (path: string) => {
    if (path === '/chat/sessions') {
      cloneCounter += 1;
      return {session_id: `clone-${cloneCounter}`};
    }
    throw new Error(`unexpected POST ${path}`);
  });
  (mockApi.put as jest.Mock).mockImplementation(async (path: string, body: any) => {
    const m = /^\/chat\/sessions\/([^/]+)$/.exec(path);
    if (m) {
      liveSessions = [
        ...liveSessions,
        {
          id: m[1],
          name: body?.name ?? 'New conversation',
          created_at: '',
          updated_at: '',
          message_count: body?.messages?.length ?? 0,
        },
      ];
      return {};
    }
    throw new Error(`unexpected PUT ${path}`);
  });
  (mockApi.delete as jest.Mock).mockImplementation(async (path: string) => {
    const m = /^\/chat\/sessions\/([^/]+)$/.exec(path);
    if (m) {
      liveSessions = liveSessions.filter(s => s.id !== m[1]);
      return {};
    }
    throw new Error(`unexpected DELETE ${path}`);
  });
}

beforeEach(() => {
  jest.clearAllMocks();
  liveSessions = [...SEED_SESSIONS];
  cloneCounter = 0;
  alertCalls = [];
  promptCb = null;
  wireStorage();
  wireApi();
  jest.spyOn(Alert, 'alert').mockImplementation(((title: any, _message: any, buttons?: any) => {
    alertCalls.push({title: String(title), buttons: (buttons ?? []) as AlertButton[]});
  }) as any);
  jest.spyOn(Alert, 'prompt').mockImplementation(((...args: any[]) => {
    promptCb = args[2] as (value: string) => void;
  }) as any);
  useChatStore.setState({sessions: [...SEED_SESSIONS], activeSessionId: 's1', messages: []});
});

async function seedFolder(name: string): Promise<string> {
  await foldersService.createFolder(name);
  const all = await foldersService.getFolders();
  return all.find(f => f.name === name)!.id;
}

function DrawerHarness() {
  const [labelFilter, setLabelFilter] = React.useState<string | null>(null);
  const [folderFilter, setFolderFilter] = React.useState<string | null>(null);
  const [showArchived, setShowArchived] = React.useState(false);
  const [labelInput, setLabelInput] = React.useState('');
  const [starredIds, setStarredIds] = React.useState<string[]>([]);
  const [sessionLabels, setSessionLabels] = React.useState<Record<string, string[]>>({});
  const [allLabels, setAllLabels] = React.useState<string[]>([]);
  const sessions = useChatStore((s: any) => s.sessions) as typeof SEED_SESSIONS;
  const [visible, setVisible] = React.useState(true);

  return (
    <ChatDrawer
      visible={visible}
      onClose={() => setVisible(false)}
      sortedActiveSessions={sessions}
      archivedSessions={[]}
      sessionLabels={sessionLabels}
      allLabels={allLabels}
      labelFilter={labelFilter}
      setLabelFilter={setLabelFilter}
      folderFilter={folderFilter}
      setFolderFilter={setFolderFilter}
      showArchived={showArchived}
      setShowArchived={setShowArchived}
      labelInput={labelInput}
      setLabelInput={setLabelInput}
      starredIds={starredIds}
      setStarredIds={setStarredIds}
      setSessionLabels={setSessionLabels}
      setAllLabels={setAllLabels}
    />
  );
}

async function renderJourney() {
  return render(<DrawerHarness />);
}

function pressAlertButton(title: string, buttonText: string) {
  const call = [...alertCalls].reverse().find(c => c.title === title);
  const button = call?.buttons.find(b => b.text === buttonText);
  if (!button || !button.onPress) {
    throw new Error(`alert button not found: ${title} / ${buttonText}`);
  }
  return act(async () => {
    button.onPress!();
  });
}

describe('journey: Mobile: Organize My Conversations', () => {
  it('creates a folder from the drawer and its chip appears', async () => {
    const {getByTestId, getAllByText} = await renderJourney();
    await waitFor(() => expect(getAllByText('Trip plans').length).toBeGreaterThan(0));

    await fireEvent.press(getByTestId('new-folder'));
    expect(promptCb).toBeTruthy();
    await act(async () => {
      promptCb!('Work');
    });

    await waitFor(async () => {
      const folders = await foldersService.getFolders();
      expect(folders.map(f => f.name)).toContain('Work');
    });
    await waitFor(() => expect(getAllByText('Work').length).toBeGreaterThan(0));
  });

  it('moves a chat into a folder, then the folder chip filters the list', async () => {
    const folderId = await seedFolder('Trips');
    const {getByTestId, getByText, queryByText} = await renderJourney();
    await waitFor(() => expect(getByText('Recipes')).toBeTruthy());

    await fireEvent(getByText('Trip plans'), 'longPress');
    await pressAlertButton('Trip plans', 'Move to Folder');
    await pressAlertButton('Move to Folder', 'Trips');

    await waitFor(async () => {
      const fid = await foldersService.getFolderOf('s1');
      expect(fid).toBe(folderId);
    });

    await fireEvent.press(getByTestId(`folder-chip-${folderId}`));
    await waitFor(() => expect(queryByText('Recipes')).toBeNull());
    expect(getByText('Trip plans')).toBeTruthy();

    await fireEvent.press(getByTestId(`folder-chip-${folderId}`));
    await waitFor(() => expect(getByText('Recipes')).toBeTruthy());
  });

  it('clones a chat — the copy is stored and becomes active, drawer closes', async () => {
    const utils = await renderJourney();
    const {getByText, queryByText} = utils;
    await waitFor(() => expect(getByText('Trip plans')).toBeTruthy());

    await fireEvent(getByText('Trip plans'), 'longPress');
    await pressAlertButton('Trip plans', 'Clone');

    await waitFor(() => expect(mockApi.post).toHaveBeenCalledWith('/chat/sessions', {}));

    const putCall = (mockApi.put as jest.Mock).mock.calls.find(
      ([path]) => typeof path === 'string' && path.startsWith('/chat/sessions/clone-'),
    );
    expect(putCall).toBeTruthy();
    expect((putCall![1] as any).name).toBe('Trip plans (Copy)');
    expect((putCall![1] as any).messages.length).toBe(1);

    // Product behavior: after cloning, the copy is loaded and the drawer closes.
    await waitFor(() => {
      const state: any = useChatStore.getState();
      expect(state.sessions.map((s: any) => s.name)).toContain('Trip plans (Copy)');
      expect(state.activeSessionId).toBe('clone-1');
    });
    await waitFor(() => expect(queryByText('Conversations')).toBeNull());

    // Original untouched in the session list.
    const names = (useChatStore.getState() as any).sessions.map((s: any) => s.name);
    expect(names).toContain('Trip plans');
    expect(names).toContain('Recipes');
  });

  it('deletes a chat from the action sheet after confirmation', async () => {
    const {getByText, queryByText} = await renderJourney();
    await waitFor(() => expect(getByText('Recipes')).toBeTruthy());

    await fireEvent(getByText('Recipes'), 'longPress');
    await pressAlertButton('Recipes', 'Delete');
    await pressAlertButton('Delete', 'Delete');

    await waitFor(() => expect(mockApi.delete).toHaveBeenCalledWith('/chat/sessions/s2'));
    await waitFor(() => expect(queryByText('Recipes')).toBeNull());
    expect(getByText('Trip plans')).toBeTruthy();
  });
});
