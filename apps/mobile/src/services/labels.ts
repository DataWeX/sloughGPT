import AsyncStorage from '@react-native-async-storage/async-storage';

const STORAGE_KEY = '@sloughgpt/session-labels';

let cache: Record<string, string[]> | null = null;

async function load(): Promise<Record<string, string[]>> {
  if (cache) return cache;
  try {
    const raw = await AsyncStorage.getItem(STORAGE_KEY);
    if (raw) {
      cache = JSON.parse(raw);
      return cache!;
    }
  } catch {}
  cache = {};
  return cache;
}

async function save(data: Record<string, string[]>): Promise<void> {
  cache = data;
  await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(data));
}

/** Get labels for a specific session. */
export async function getLabels(sessionId: string): Promise<string[]> {
  const data = await load();
  return data[sessionId] || [];
}

/** Set labels for a specific session (replaces all). */
export async function setLabels(sessionId: string, labels: string[]): Promise<void> {
  const data = await load();
  const unique = [...new Set(labels.map(l => l.trim()).filter(Boolean))];
  if (unique.length > 0) {
    data[sessionId] = unique;
  } else {
    delete data[sessionId];
  }
  await save(data);
}

/** Add a label to a session. */
export async function addLabel(sessionId: string, label: string): Promise<void> {
  const existing = await getLabels(sessionId);
  const trimmed = label.trim();
  if (trimmed && !existing.includes(trimmed)) {
    await setLabels(sessionId, [...existing, trimmed]);
  }
}

/** Remove a label from a session. */
export async function removeLabel(sessionId: string, label: string): Promise<void> {
  const existing = await getLabels(sessionId);
  await setLabels(sessionId, existing.filter(l => l !== label));
}

/** Get all distinct labels across all sessions. */
export async function getAllDistinctLabels(): Promise<string[]> {
  const data = await load();
  const all = new Set<string>();
  for (const labels of Object.values(data)) {
    for (const label of labels) {
      all.add(label);
    }
  }
  return [...all].sort();
}

/** Reset the in-memory cache. Used in tests. */
export function _resetCache(): void {
  cache = null;
}
