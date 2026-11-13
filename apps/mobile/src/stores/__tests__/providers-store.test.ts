import AsyncStorage from '@react-native-async-storage/async-storage';

jest.mock('../../services/haptics');
jest.mock('../../services/sounds');
jest.mock('../../services/toast');

const STORAGE_KEY = '@sloughgpt/providers';
const ACTIVE_KEY = '@sloughgpt/active_provider';

// eslint-disable-next-line @typescript-eslint/no-var-requires
const {useProvidersStore} = require('../providers-store');

// Capture default state before any mutations
const DEFAULT_PROVIDERS = JSON.parse(JSON.stringify(useProvidersStore.getState().providers));

beforeEach(async () => {
  jest.clearAllMocks();
  (AsyncStorage.getItem as jest.Mock).mockResolvedValue(null);
  (AsyncStorage.setItem as jest.Mock).mockResolvedValue(undefined);

  // Reset store to deep-cloned defaults
  useProvidersStore.setState({
    providers: JSON.parse(JSON.stringify(DEFAULT_PROVIDERS)),
    activeProviderId: null,
  });
  await new Promise(r => setTimeout(r, 5));
});

describe('providers-store', () => {
  describe('initial state', () => {
    it('has all 9 provider configs', () => {
      const s = useProvidersStore.getState();
      expect(Object.keys(s.providers)).toHaveLength(9);
      expect(s.providers.openai).toBeDefined();
      expect(s.providers.anthropic).toBeDefined();
      expect(s.providers.google).toBeDefined();
      expect(s.providers.mistral).toBeDefined();
      expect(s.providers.groq).toBeDefined();
      expect(s.providers.together).toBeDefined();
      expect(s.providers.deepseek).toBeDefined();
      expect(s.providers.openrouter).toBeDefined();
      expect(s.providers.custom).toBeDefined();
    });

    it('no active provider by default', () => {
      expect(useProvidersStore.getState().activeProviderId).toBeNull();
    });

    it('all providers have empty API keys', () => {
      const s = useProvidersStore.getState();
      for (const p of Object.values(s.providers)) {
        expect(p.apiKey).toBe('');
      }
    });

    it('all providers have default base URLs', () => {
      const s = useProvidersStore.getState();
      expect(s.providers.openai.baseUrl).toBe('https://api.openai.com/v1');
      expect(s.providers.anthropic.baseUrl).toBe('https://api.anthropic.com/v1');
      expect(s.providers.google.baseUrl).toBe('https://generativelanguage.googleapis.com/v1beta');
    });
  });

  describe('setActiveProvider', () => {
    it('sets the active provider', async () => {
      await useProvidersStore.getState().setActiveProvider('openai');
      expect(useProvidersStore.getState().activeProviderId).toBe('openai');
    });

    it('clears the active provider', async () => {
      await useProvidersStore.getState().setActiveProvider('openai');
      await useProvidersStore.getState().setActiveProvider(null);
      expect(useProvidersStore.getState().activeProviderId).toBeNull();
    });

    it('persists to AsyncStorage', async () => {
      await useProvidersStore.getState().setActiveProvider('anthropic');
      expect(AsyncStorage.setItem).toHaveBeenCalledWith(ACTIVE_KEY, 'anthropic');
    });
  });

  describe('setApiKey', () => {
    it('sets the API key for a provider', async () => {
      await useProvidersStore.getState().setApiKey('openai', 'sk-test-123');
      expect(useProvidersStore.getState().providers.openai.apiKey).toBe('sk-test-123');
    });

    it('does not affect other providers', async () => {
      await useProvidersStore.getState().setApiKey('openai', 'sk-test-123');
      expect(useProvidersStore.getState().providers.anthropic.apiKey).toBe('');
    });

    it('persists to AsyncStorage', async () => {
      await useProvidersStore.getState().setApiKey('openai', 'sk-test-123');
      expect(AsyncStorage.setItem).toHaveBeenCalled();
    });
  });

  describe('setBaseUrl', () => {
    it('sets the base URL for a provider', async () => {
      await useProvidersStore.getState().setBaseUrl('custom', 'http://localhost:11434/v1');
      expect(useProvidersStore.getState().providers.custom.baseUrl).toBe('http://localhost:11434/v1');
    });
  });

  describe('setDefaultModel', () => {
    it('sets the default model for a provider', async () => {
      await useProvidersStore.getState().setDefaultModel('openai', 'gpt-4o');
      expect(useProvidersStore.getState().providers.openai.defaultModel).toBe('gpt-4o');
    });
  });

  describe('toggleProvider', () => {
    it('disables a provider', async () => {
      await useProvidersStore.getState().toggleProvider('openai', false);
      expect(useProvidersStore.getState().providers.openai.enabled).toBe(false);
    });

    it('enables a provider', async () => {
      await useProvidersStore.getState().toggleProvider('openai', false);
      await useProvidersStore.getState().toggleProvider('openai', true);
      expect(useProvidersStore.getState().providers.openai.enabled).toBe(true);
    });
  });

  describe('resetProvider', () => {
    it('resets provider to defaults but keeps API key', async () => {
      await useProvidersStore.getState().setApiKey('openai', 'sk-test');
      await useProvidersStore.getState().setBaseUrl('openai', 'http://custom');
      await useProvidersStore.getState().setDefaultModel('openai', 'gpt-4o');
      await useProvidersStore.getState().resetProvider('openai');

      const p = useProvidersStore.getState().providers.openai;
      expect(p.baseUrl).toBe('https://api.openai.com/v1');
      expect(p.defaultModel).toBe('gpt-4o-mini');
      expect(p.apiKey).toBe('sk-test'); // preserved
    });
  });

  describe('getActiveConfig', () => {
    it('returns null when no provider is active', () => {
      expect(useProvidersStore.getState().getActiveConfig()).toBeNull();
    });

    it('returns the active provider config', async () => {
      await useProvidersStore.getState().setApiKey('openai', 'sk-test');
      await useProvidersStore.getState().setActiveProvider('openai');
      const config = useProvidersStore.getState().getActiveConfig();
      expect(config).not.toBeNull();
      expect(config!.id).toBe('openai');
      expect(config!.apiKey).toBe('sk-test');
    });
  });

  describe('hasAnyApiKey', () => {
    it('returns false when no keys configured', () => {
      expect(useProvidersStore.getState().hasAnyApiKey()).toBe(false);
    });

    it('returns true when at least one key is set', async () => {
      await useProvidersStore.getState().setApiKey('groq', 'gsk_test');
      expect(useProvidersStore.getState().hasAnyApiKey()).toBe(true);
    });
  });
});
