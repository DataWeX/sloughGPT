/**
 * File upload service — pick documents and send to chat.
 * Supports text files, code files, and images.
 */

let DocumentPicker: any;
try {
  DocumentPicker = require('expo-document-picker');
} catch {
  DocumentPicker = null;
}

export interface PickedFile {
  name: string;
  uri: string;
  mimeType: string;
  size: number;
  content?: string;
}

const TEXT_TYPES = [
  'text/plain', 'text/markdown', 'text/csv', 'text/html', 'text/css',
  'application/json', 'application/xml', 'application/javascript',
  'application/typescript', 'application/x-python',
];

const CODE_EXTENSIONS = [
  '.py', '.js', '.ts', '.tsx', '.jsx', '.json', '.md', '.txt',
  '.css', '.html', '.xml', '.yaml', '.yml', '.toml', '.sh',
  '.sql', '.r', '.java', '.cpp', '.c', '.h', '.go', '.rs',
  '.swift', '.kt', '.rb', '.php', '.lua', '.zig',
];

export async function pickDocument(): Promise<PickedFile | null> {
  if (!DocumentPicker) {
    throw new Error('Document picker not available. Install expo-document-picker.');
  }

  const result = await DocumentPicker.getDocumentAsync({
    type: '*/*',
    copyToCacheDirectory: true,
    multiple: false,
  });

  if (result.canceled || !result.assets?.[0]) return null;

  const asset = result.assets[0];
  return {
    name: asset.name,
    uri: asset.uri,
    mimeType: asset.mimeType || 'application/octet-stream',
    size: asset.size || 0,
  };
}

export function isTextFile(file: PickedFile): boolean {
  if (TEXT_TYPES.includes(file.mimeType)) return true;
  const ext = '.' + file.name.split('.').pop()?.toLowerCase();
  return CODE_EXTENSIONS.includes(ext);
}

export function isImageFile(file: PickedFile): boolean {
  return file.mimeType.startsWith('image/');
}

export function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
