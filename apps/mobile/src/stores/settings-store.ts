import {create} from 'zustand';
import AsyncStorage from '@react-native-async-storage/async-storage';
import type {ThemeMode} from '../types';

interface SettingsState {
  theme: ThemeMode;
  temperature: number;
  maxTokens: number;
  memoryContext: string;
  apiUrl: string;
  setTheme: (theme: ThemeMode) => void;
  update: (partial: Partial<SettingsState>) => void;
  reset: () => void;
}

const STORAGE_KEY = '@sloughgpt/settings';

const defaults: Omit<SettingsState, 'setTheme' | 'update' | 'reset'> = {
  theme: 'system',
  temperature: 0.8,
  maxTokens: 256,
  memoryContext: '',
  apiUrl: 'http://localhost:8000',
};

async function loadSettings() {
  try {
    const raw = await AsyncStorage.getItem(STORAGE_KEY);
    if (raw) return JSON.parse(raw);
  } catch {}
  return {};
}

const saved = await loadSettings();

export const useSettingsStore = create<SettingsState>((set, get) => ({
  ...defaults,
  ...saved,
  setTheme: theme => set({theme}),
  update: partial => {
    set(partial);
    const state = get();
    const {setTheme, update, reset, ...persist} = state;
    AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(persist));
  },
  reset: () => {
    set(defaults);
    AsyncStorage.removeItem(STORAGE_KEY);
  },
}));
