import AsyncStorage from '@react-native-async-storage/async-storage';

const STORAGE_KEY = '@sloughgpt/knowledge';

export interface KnowledgeFact {
  id: string;
  content: string;
  source: 'user' | 'assistant';
  messageId: string;
  savedAt: number;
}

let cached: KnowledgeFact[] | null = null;

async function load(): Promise<KnowledgeFact[]> {
  if (cached) return cached;
  try {
    const raw = await AsyncStorage.getItem(STORAGE_KEY);
    cached = raw ? JSON.parse(raw) : [];
  } catch {
    cached = [];
  }
  return cached!;
}

async function save(facts: KnowledgeFact[]): Promise<void> {
  cached = facts;
  await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(facts));
}

export async function saveToKnowledge(
  content: string,
  source: 'user' | 'assistant',
  messageId: string,
): Promise<boolean> {
  const all = await load();
  if (all.some(f => f.messageId === messageId)) return false;
  const fact: KnowledgeFact = {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    content,
    source,
    messageId,
    savedAt: Date.now(),
  };
  all.unshift(fact);
  await save(all);
  return true;
}

export async function getKnowledge(): Promise<KnowledgeFact[]> {
  return load();
}

export async function removeKnowledge(id: string): Promise<void> {
  const all = await load();
  await save(all.filter(f => f.id !== id));
}

export async function clearKnowledge(): Promise<void> {
  cached = null;
  await AsyncStorage.removeItem(STORAGE_KEY);
}

export async function getKnowledgeForMessage(messageId: string): Promise<KnowledgeFact | null> {
  const all = await load();
  return all.find(f => f.messageId === messageId) ?? null;
}

export function _resetCache(): void {
  cached = null;
}
