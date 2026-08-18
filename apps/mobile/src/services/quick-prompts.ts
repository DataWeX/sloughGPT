/**
 * Quick prompts — saved prompt templates for fast access.
 * Stored in AsyncStorage, user can add/edit/delete custom prompts.
 */

import AsyncStorage from '@react-native-async-storage/async-storage';

const STORAGE_KEY = '@sloughgpt/quick-prompts';

export interface QuickPrompt {
  id: string;
  title: string;
  prompt: string;
  category: 'general' | 'coding' | 'writing' | 'analysis' | 'custom';
  createdAt: number;
}

const DEFAULT_PROMPTS: QuickPrompt[] = [
  {
    id: 'explain',
    title: 'Explain concept',
    prompt: 'Explain the concept of {topic} in simple terms with examples.',
    category: 'general',
    createdAt: Date.now(),
  },
  {
    id: 'debug',
    title: 'Debug code',
    prompt: 'Help me debug this code. Explain what\'s wrong and how to fix it:\n\n```\n{code}\n```',
    category: 'coding',
    createdAt: Date.now(),
  },
  {
    id: 'review',
    title: 'Code review',
    prompt: 'Review this code for performance, security, and best practices:\n\n```\n{code}\n```',
    category: 'coding',
    createdAt: Date.now(),
  },
  {
    id: 'write',
    title: 'Write email',
    prompt: 'Write a professional email about: {topic}',
    category: 'writing',
    createdAt: Date.now(),
  },
  {
    id: 'summarize',
    title: 'Summarize',
    prompt: 'Summarize the following into key points:\n\n{text}',
    category: 'analysis',
    createdAt: Date.now(),
  },
  {
    id: 'translate',
    title: 'Translate',
    prompt: 'Translate the following to {language}:\n\n{text}',
    category: 'general',
    createdAt: Date.now(),
  },
];

let cached: QuickPrompt[] | null = null;

export function _resetCache() {
  cached = null;
}

async function load(): Promise<QuickPrompt[]> {
  if (cached) return cached;
  try {
    const raw = await AsyncStorage.getItem(STORAGE_KEY);
    if (raw) {
      cached = JSON.parse(raw);
      return cached!;
    }
  } catch {}
  // First time — seed with defaults (clone to avoid mutation)
  cached = DEFAULT_PROMPTS.map(p => ({...p}));
  await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(cached));
  return cached;
}

async function save(prompts: QuickPrompt[]): Promise<void> {
  cached = prompts;
  await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(prompts));
}

export async function getQuickPrompts(): Promise<QuickPrompt[]> {
  return load();
}

export async function getQuickPromptsByCategory(category: string): Promise<QuickPrompt[]> {
  const all = await load();
  if (category === 'all') return all;
  return all.filter(p => p.category === category);
}

export async function addQuickPrompt(title: string, prompt: string, category: QuickPrompt['category'] = 'custom'): Promise<QuickPrompt> {
  const all = await load();
  const newPrompt: QuickPrompt = {
    id: `custom-${Date.now()}`,
    title,
    prompt,
    category,
    createdAt: Date.now(),
  };
  all.push(newPrompt);
  await save(all);
  return newPrompt;
}

export async function updateQuickPrompt(id: string, updates: Partial<Pick<QuickPrompt, 'title' | 'prompt' | 'category'>>): Promise<void> {
  const all = await load();
  const idx = all.findIndex(p => p.id === id);
  if (idx !== -1) {
    all[idx] = {...all[idx], ...updates};
    await save(all);
  }
}

export async function deleteQuickPrompt(id: string): Promise<void> {
  const all = await load();
  await save(all.filter(p => p.id !== id));
}

/**
 * Fill template placeholders: {topic}, {code}, {text}, {language}
 * Returns the prompt with filled values or the raw prompt if no params.
 */
export function fillTemplate(template: string, params?: Record<string, string>): string {
  if (!params) return template;
  return template.replace(/\{(\w+)\}/g, (_, key) => params[key] || `{${key}}`);
}
