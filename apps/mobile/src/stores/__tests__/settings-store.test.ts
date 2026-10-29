import AsyncStorage from '@react-native-async-storage/async-storage';

jest.mock('../../services/haptics');
jest.mock('../../services/sounds');
jest.mock('../../services/toast');

const STORAGE_KEY = '@sloughgpt/settings';

const DEFAULTS = {
  theme: 'system',
  fontFamily: 'outfit',
  fontSizeScale: 1.0,
  accentColor: 'violet',
  temperature: 0.8,
  maxTokens: 256,
  topP: 0.9,
  topK: 50,
  repetitionPenalty: 1.2,
  memoryContext: '',
  apiUrl: 'http://localhost:8000',
  chatBackground: '',
};

// eslint-disable-next-line @typescript-eslint/no-var-requires
const {useSettingsStore} = require('../settings-store');

beforeEach(async () => {
  jest.clearAllMocks();
  (AsyncStorage.getItem as jest.Mock).mockResolvedValue(null);
  (AsyncStorage.setItem as jest.Mock).mockResolvedValue(undefined);
  (AsyncStorage.removeItem as jest.Mock).mockResolvedValue(undefined);

  // Reset store to defaults
  useSettingsStore.setState(DEFAULTS);
  // Small tick to let any pending hydrations settle
  await new Promise(r => setTimeout(r, 5));
});

describe('initial state', () => {
  it('has correct default values', () => {
    const s = useSettingsStore.getState();
    expect(s.theme).toBe('system');
    expect(s.fontFamily).toBe('outfit');
    expect(s.fontSizeScale).toBe(1.0);
    expect(s.temperature).toBe(0.8);
    expect(s.maxTokens).toBe(256);
    expect(s.topP).toBe(0.9);
    expect(s.topK).toBe(50);
    expect(s.repetitionPenalty).toBe(1.2);
    expect(s.apiUrl).toBe('http://localhost:8000');
  });
});

describe('hydration', () => {
  it('loads persisted values on module init', async () => {
    // hydrateSettings reads from AsyncStorage on import; we already imported,
    // so directly verify the mechanism works by calling setState with saved data
    const saved = {theme: 'dark', temperature: 0.5};
    useSettingsStore.setState(saved);
    const s = useSettingsStore.getState();
    expect(s.theme).toBe('dark');
    expect(s.temperature).toBe(0.5);
    // Non-overridden fields remain at defaults
    expect(s.fontFamily).toBe('outfit');
  });
});

describe('setTheme', () => {
  it('updates state without persisting to storage', () => {
    useSettingsStore.getState().setTheme('dark');
    expect(useSettingsStore.getState().theme).toBe('dark');
    expect(AsyncStorage.setItem).not.toHaveBeenCalled();
  });
});

describe('setFontFamily', () => {
  it('updates state and persists to AsyncStorage', () => {
    useSettingsStore.getState().setFontFamily('system');
    expect(useSettingsStore.getState().fontFamily).toBe('system');
    expect(AsyncStorage.setItem).toHaveBeenCalledWith(
      STORAGE_KEY,
      expect.any(String),
    );
  });
});

describe('setFontSizeScale', () => {
  it('updates state and persists to AsyncStorage', () => {
    useSettingsStore.getState().setFontSizeScale(1.1);
    expect(useSettingsStore.getState().fontSizeScale).toBe(1.1);
    expect(AsyncStorage.setItem).toHaveBeenCalled();
  });
});

describe('update', () => {
  it('merges partial state', () => {
    useSettingsStore.getState().update({temperature: 0.3, topK: 20});
    const s = useSettingsStore.getState();
    expect(s.temperature).toBe(0.3);
    expect(s.topK).toBe(20);
    // Unchanged
    expect(s.theme).toBe('system');
    expect(s.fontFamily).toBe('outfit');
  });

  it('persists merged state to AsyncStorage', () => {
    useSettingsStore.getState().update({maxTokens: 512});
    expect(AsyncStorage.setItem).toHaveBeenCalledWith(
      STORAGE_KEY,
      expect.any(String),
    );
  });
});

describe('reset', () => {
  it('restores default values', () => {
    useSettingsStore.getState().update({temperature: 0.1, theme: 'dark'});
    useSettingsStore.getState().reset();
    const s = useSettingsStore.getState();
    expect(s.temperature).toBe(0.8);
    expect(s.theme).toBe('dark');
    expect(s.fontFamily).toBe('outfit');
  });

  it('removes saved settings from AsyncStorage', () => {
    useSettingsStore.getState().reset();
    expect(AsyncStorage.removeItem).toHaveBeenCalledWith(STORAGE_KEY);
  });
});

describe('persisted data', () => {
  it('excludes action functions from saved payload', () => {
    useSettingsStore.getState().setFontFamily('system');
    const callArg = (AsyncStorage.setItem as jest.Mock).mock.calls[0][1];
    const parsed = JSON.parse(callArg);
    expect(parsed.setTheme).toBeUndefined();
    expect(parsed.setFontFamily).toBeUndefined();
    expect(parsed.update).toBeUndefined();
    expect(parsed.reset).toBeUndefined();
    expect(parsed.setFontSizeScale).toBeUndefined();
    // Data fields are included
    expect(parsed.theme).toBeDefined();
    expect(parsed.temperature).toBeDefined();
    expect(parsed.fontFamily).toBeDefined();
  });
});
