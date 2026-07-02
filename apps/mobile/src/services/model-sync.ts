/**
 * Model sync service.
 *
 * After activity classifier training completes on the server, downloads
 * the updated model.npz weights for local caching. The weights can be
 * used for on-device prediction without a server round-trip.
 *
 * Flow:
 *   1. Server finishes training → model.npz saved to disk on the server
 *   2. This service downloads model.npz via /activity/model/download
 *      (or fetches it from a known path)
 *   3. Caches it locally in AsyncStorage
 *   4. Reports status so the UI can show "Model updated"
 *
 * Requires a backend endpoint to serve the model file.
 * For now, we check /activity/status.model_loaded as a proxy.
 */

import AsyncStorage from '@react-native-async-storage/async-storage';
import {getApiUrl} from './api-client';

const MODEL_CACHE_KEY = '@sloughgpt/activity_model_cached';
const MODEL_VERSION_KEY = '@sloughgpt/activity_model_version';

export interface ModelSyncStatus {
  cached: boolean;
  version: number;
  lastSync: number | null;
  fileSize: number | null;
}

/** Try to download the trained model from the server and cache it. */
export async function syncModel(): Promise<boolean> {
  try {
    const baseUrl = await getApiUrl();

    // Check if model exists on server
    const statusRes = await fetch(`${baseUrl}/activity/status`);
    if (!statusRes.ok) return false;
    const status = await statusRes.json();
    if (!status.model_loaded) return false;

    // Try to download model.npz from the server
    const modelUrl = `${baseUrl}/activity/model`;
    const modelRes = await fetch(modelUrl);
    if (!modelRes.ok) {
      return false;
    }

    const blob = await modelRes.blob();
    const reader = new FileReader();
    const base64 = await new Promise<string>((resolve, reject) => {
      reader.onload = () => resolve(reader.result as string);
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });

    await AsyncStorage.setItem(MODEL_CACHE_KEY, base64);
    await AsyncStorage.setItem(MODEL_VERSION_KEY, String(Date.now()));
    return true;
  } catch {
    return false;
  }
}

/** Check whether we have a cached model. */
export async function getModelSyncStatus(): Promise<ModelSyncStatus> {
  try {
    const cached = await AsyncStorage.getItem(MODEL_CACHE_KEY);
    const version = await AsyncStorage.getItem(MODEL_VERSION_KEY);
    return {
      cached: cached === 'true' || (cached !== null && cached.startsWith('data:')),
      version: version ? Number(version) : 0,
      lastSync: version ? Number(version) : null,
      fileSize: cached !== null ? cached.length : null,
    };
  } catch {
    return {cached: false, version: 0, lastSync: null, fileSize: null};
  }
}

/** Clear the cached model. */
export async function clearCachedModel(): Promise<void> {
  await AsyncStorage.multiRemove([MODEL_CACHE_KEY, MODEL_VERSION_KEY]);
}

/** Get cached model as base64 data URL, or null. */
export async function getCachedModelData(): Promise<string | null> {
  try {
    return await AsyncStorage.getItem(MODEL_CACHE_KEY);
  } catch {
    return null;
  }
}
