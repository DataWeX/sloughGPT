import {Platform, Share} from 'react-native';
import RNFS from 'react-native-fs';

export async function downloadJson(data: unknown, filename: string): Promise<boolean> {
  try {
    const json = JSON.stringify(data, null, 2);
    const path = `${RNFS.DocumentDirectoryPath}/${filename}`;
    await RNFS.writeFile(path, json);

    if (Platform.OS === 'ios') {
      const result = await Share.share({
        url: `file://${path}`,
        title: filename,
      });
      return result.action !== Share.dismissedAction;
    } else {
      const result = await Share.share({
        title: filename,
        message: json,
      });
      return result.action !== Share.dismissedAction;
    }
  } catch {
    return false;
  }
}

export async function downloadText(content: string, filename: string): Promise<boolean> {
  try {
    const path = `${RNFS.DocumentDirectoryPath}/${filename}`;
    await RNFS.writeFile(path, content);

    if (Platform.OS === 'ios') {
      const result = await Share.share({
        url: `file://${path}`,
        title: filename,
      });
      return result.action !== Share.dismissedAction;
    } else {
      const result = await Share.share({
        title: filename,
        message: content,
      });
      return result.action !== Share.dismissedAction;
    }
  } catch {
    return false;
  }
}

export async function downloadBinary(data: ArrayBuffer, filename: string, _mimeType: string): Promise<boolean> {
  try {
    const path = `${RNFS.DocumentDirectoryPath}/${filename}`;
    const bytes = new Uint8Array(data);
    let binary = '';
    for (let i = 0; i < bytes.byteLength; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    const base64 = global.btoa(binary);
    await RNFS.writeFile(path, base64);

    if (Platform.OS === 'ios') {
      const result = await Share.share({
        url: `file://${path}`,
        title: filename,
      });
      return result.action !== Share.dismissedAction;
    } else {
      const result = await Share.share({
        title: filename,
        message: `Shared ${filename}`,
      });
      return result.action !== Share.dismissedAction;
    }
  } catch {
    return false;
  }
}

export function todayDateString(): string {
  return new Date().toISOString().slice(0, 10);
}

export function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
}

export function formatDuration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  const h = Math.floor(seconds / 3600);
  const m = Math.round((seconds % 3600) / 60);
  return `${h}h ${m}m`;
}
