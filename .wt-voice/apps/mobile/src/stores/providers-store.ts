/**
 * Providers store — manages API keys and settings for third-party inference providers.
 *
 * Persists to AsyncStorage. Each provider config includes baseUrl, apiKey,
 * defaultModel, and enabled state. The store also tracks the active provider
 * (which provider handles remote inference when the hybrid store routes to 'remote').
 */

import {create} from 'zustand';
import AsyncStorage from '@react-native-async-storage/async-storage';
import {
  PROVIDER_REGISTRY,
  type ProviderId,
  type ProviderConfig,
} from '../types/providers';

const STORAGE_KEY = '@sloughgpt/providers';
const ACTIVE_KEY = '@sloughgpt/active_provider';

interface ProvidersState {
  /** Map of provider ID → config. */
  providers: Record<ProviderId, ProviderConfig>;
  /** Currently active provider for remote inference. */
  activeProviderId: ProviderId | null;
  /** Whether hydration from AsyncStorage is complete. */
  hydrated: boolean;

  /** Set the active provider for remote inference. */
  setActiveProvider: (id: ProviderId | null) => Promise<void>;
  /** Update a single provider's config (partial merge). */
  updateProvider: (id: ProviderId, patch: Partial<ProviderConfig>) => Promise<void>;
  /** Toggle a provider on/off. */
  toggleProvider: (id: ProviderId, enabled: boolean) => Promise<void>;
  /** Set API key for a provider. */
  setApiKey: (id: ProviderId, apiKey: string) => Promise<void>;
  /** Set base URL for a provider. */
  setBaseUrl: (id: ProviderId, baseUrl: string) => Promise<void>;
  /** Set default model for a provider. */
  setDefaultModel: (id: ProviderId, model: string) => Promise<void>;
  /** Reset a provider to its registry defaults. */
  resetProvider: (id: ProviderId) => Promise<void>;
  /** Get config for the currently active provider (or null). */
  getActiveConfig: () => ProviderConfig | null;
  /** Check if any provider has an API key configured. */
  hasAnyApiKey: () => boolean;
}

function _defaultConfigs(): Record<ProviderId, ProviderConfig> {
  const out = {} as Record<ProviderId, ProviderConfig>;
  for (const [id, def] of Object.entries(PROVIDER_REGISTRY) as [ProviderId, typeof PROVIDER_REGISTRY[ProviderId]][]) {
    out[id] = {...def, apiKey: ''};
  }
  return out;
}

let _loaded = false;

async function _hydrate(store: {setState: (s: Partial<ProvidersState>) => void}) {
  if (_loaded) return;
  _loaded = true;
  try {
    const [rawProviders, rawActive] = await Promise.all([
      AsyncStorage.getItem(STORAGE_KEY),
      AsyncStorage.getItem(ACTIVE_KEY),
    ]);

    const defaults = _defaultConfigs();
    let providers = defaults;

    if (rawProviders) {
      const stored = JSON.parse(rawProviders) as Record<ProviderId, ProviderConfig>;
      // Merge stored over defaults — new providers added via app update are preserved
      for (const key of Object.keys(defaults) as ProviderId[]) {
        if (stored[key]) {
          providers[key] = {...defaults[key], ...stored[key]};
        }
      }
    }

    const activeProviderId = rawActive as ProviderId | null;
    store.setState({providers, activeProviderId, hydrated: true});
  } catch {
    store.setState({hydrated: true});
  }
}

async function _persist(state: {providers: Record<ProviderId, ProviderConfig>; activeProviderId: ProviderId | null}) {
  try {
    await Promise.all([
      AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(state.providers)),
      AsyncStorage.setItem(ACTIVE_KEY, state.activeProviderId || ''),
    ]);
  } catch {}
}

export const useProvidersStore = create<ProvidersState>((set, get) => ({
  providers: _defaultConfigs(),
  activeProviderId: null,
  hydrated: false,

  setActiveProvider: async id => {
    set({activeProviderId: id});
    await _persist({...get(), activeProviderId: id});
  },

  updateProvider: async (id, patch) => {
    set(s => ({
      providers: {
        ...s.providers,
        [id]: {...s.providers[id], ...patch},
      },
    }));
    await _persist(get());
  },

  toggleProvider: async (id, enabled) => {
    set(s => ({
      providers: {
        ...s.providers,
        [id]: {...s.providers[id], enabled},
      },
    }));
    await _persist(get());
  },

  setApiKey: async (id, apiKey) => {
    set(s => ({
      providers: {
        ...s.providers,
        [id]: {...s.providers[id], apiKey},
      },
    }));
    await _persist(get());
  },

  setBaseUrl: async (id, baseUrl) => {
    set(s => ({
      providers: {
        ...s.providers,
        [id]: {...s.providers[id], baseUrl},
      },
    }));
    await _persist(get());
  },

  setDefaultModel: async (id, model) => {
    set(s => ({
      providers: {
        ...s.providers,
        [id]: {...s.providers[id], defaultModel: model},
      },
    }));
    await _persist(get());
  },

  resetProvider: async id => {
    const def = PROVIDER_REGISTRY[id];
    set(s => ({
      providers: {
        ...s.providers,
        [id]: {...def, apiKey: s.providers[id]?.apiKey || ''},
      },
    }));
    await _persist(get());
  },

  getActiveConfig: () => {
    const {activeProviderId, providers} = get();
    if (!activeProviderId) return null;
    return providers[activeProviderId] || null;
  },

  hasAnyApiKey: () => {
    const {providers} = get();
    return Object.values(providers).some(p => p.apiKey.length > 0);
  },
}));

// Hydrate on import
_hydrate(useProvidersStore as any);
