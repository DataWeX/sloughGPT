import {create} from 'zustand';
import AsyncStorage from '@react-native-async-storage/async-storage';
import type {ThemeMode} from '../types';

export type FontFamilyOption = 'system' | 'dm-sans';
export type FontSizeScale = 0.85 | 0.925 | 1.0 | 1.1 | 1.2;

interface SettingsState {
  theme: ThemeMode;
  fontFamily: FontFamilyOption;
  fontSizeScale: FontSizeScale;
  temperature: number;
  maxTokens: number;
  topP: number;
  topK: number;
  repetitionPenalty: number;
  memoryContext: string;
  apiUrl: string;
  chatBackground: string;
  setTheme: (theme: ThemeMode) => void;
  setFontFamily: (family: FontFamilyOption) => void;
  setFontSizeScale: (scale: FontSizeScale) => void;
  update: (partial: Partial<SettingsState>) => void;
  reset: () => void;
}

const STORAGE_KEY = '@sloughgpt/settings';

const defaults: Omit<SettingsState, 'setTheme' | 'setFontFamily' | 'setFontSizeScale' | 'update' | 'reset'> = {
  theme: 'system',
  fontFamily: 'dm-sans',
  fontSizeScale: 1.0,
  temperature: 0.8,
  maxTokens: 256,
  topP: 0.9,
  topK: 50,
  repetitionPenalty: 1.2,
  memoryContext: '',
  apiUrl: 'http://localhost:8000',
  chatBackground: '',
};

let _loaded = false;

async function hydrateSettings(store: {setState: (s: Partial<SettingsState>) => void}) {
  if (_loaded) return;
  _loaded = true;
  try {
    const raw = await AsyncStorage.getItem(STORAGE_KEY);
    if (raw) store.setState(JSON.parse(raw));
  } catch {}
}

export const useSettingsStore = create<SettingsState>((set, get) => ({
  ...defaults,
  setTheme: theme => set({theme}),
  setFontFamily: fontFamily => {
    set({fontFamily});
    const state = get();
    const {setTheme, setFontFamily, setFontSizeScale, update, reset, ...persist} = state;
    AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(persist));
  },
  setFontSizeScale: fontSizeScale => {
    set({fontSizeScale});
    const state = get();
    const {setTheme, setFontFamily, setFontSizeScale, update, reset, ...persist} = state;
    AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(persist));
  },
  update: partial => {
    set(partial);
    const state = get();
    const {setTheme, setFontFamily, setFontSizeScale, update, reset, ...persist} = state;
    AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(persist));
  },
  reset: () => {
    set(defaults);
    AsyncStorage.removeItem(STORAGE_KEY);
  },
}));

hydrateSettings(useSettingsStore);
