/**
 * Session folders — organize conversations into named groups.
 * Stored in AsyncStorage: {folders: Folder[], assignments: Record<sessionId, folderId>}.
 * Same cache pattern as stars.ts / labels.ts.
 */

import AsyncStorage from '@react-native-async-storage/async-storage';

const STORAGE_KEY = '@sloughgpt/session-folders';

export interface Folder {
  id: string;
  name: string;
  createdAt: number;
}

interface FolderStore {
  folders: Folder[];
  assignments: Record<string, string>;
}

let cache: FolderStore | null = null;

async function load(): Promise<FolderStore> {
  if (cache) return cache;
  try {
    const raw = await AsyncStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      cache = {
        folders: Array.isArray(parsed.folders) ? parsed.folders : [],
        assignments: parsed.assignments && typeof parsed.assignments === 'object' ? parsed.assignments : {},
      };
      return cache!;
    }
  } catch {}
  cache = {folders: [], assignments: {}};
  return cache;
}

async function save(data: FolderStore): Promise<void> {
  cache = data;
  await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(data));
}

/** All folders, ordered by creation. */
export async function getFolders(): Promise<Folder[]> {
  const data = await load();
  return [...data.folders].sort((a, b) => a.createdAt - b.createdAt);
}

/** Create a folder. Rejects empty or duplicate (case-insensitive) names. */
export async function createFolder(name: string): Promise<Folder> {
  const trimmed = name.trim();
  if (!trimmed) {
    throw new Error('Folder name cannot be empty');
  }
  const data = await load();
  if (data.folders.some(f => f.name.toLowerCase() === trimmed.toLowerCase())) {
    throw new Error(`Folder "${trimmed}" already exists`);
  }
  const folder: Folder = {
    id: `folder-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    name: trimmed,
    createdAt: Date.now(),
  };
  data.folders.push(folder);
  await save(data);
  return folder;
}

/** Rename a folder. Throws for unknown id. */
export async function renameFolder(folderId: string, name: string): Promise<void> {
  const trimmed = name.trim();
  if (!trimmed) {
    throw new Error('Folder name cannot be empty');
  }
  const data = await load();
  const folder = data.folders.find(f => f.id === folderId);
  if (!folder) {
    throw new Error('Folder not found');
  }
  folder.name = trimmed;
  await save(data);
}

/** Delete a folder and clear assignments pointing at it. */
export async function deleteFolder(folderId: string): Promise<void> {
  const data = await load();
  data.folders = data.folders.filter(f => f.id !== folderId);
  for (const [sessionId, fid] of Object.entries(data.assignments)) {
    if (fid === folderId) {
      delete data.assignments[sessionId];
    }
  }
  await save(data);
}

/** Move a session into a folder, or unfile it with null. Throws for unknown folder id. */
export async function moveSessionToFolder(
  sessionId: string,
  folderId: string | null,
): Promise<void> {
  const data = await load();
  if (folderId === null) {
    delete data.assignments[sessionId];
    await save(data);
    return;
  }
  if (!data.folders.some(f => f.id === folderId)) {
    throw new Error('Folder not found');
  }
  data.assignments[sessionId] = folderId;
  await save(data);
}

/** Folder id a session belongs to, or null. */
export async function getFolderOf(sessionId: string): Promise<string | null> {
  const data = await load();
  return data.assignments[sessionId] || null;
}

/** Session ids assigned to a folder. */
export async function getSessionsInFolder(folderId: string): Promise<string[]> {
  const data = await load();
  return Object.entries(data.assignments)
    .filter(([, fid]) => fid === folderId)
    .map(([sessionId]) => sessionId);
}

/** Reset the in-memory cache. Used in tests. */
export function _resetCache(): void {
  cache = null;
}
