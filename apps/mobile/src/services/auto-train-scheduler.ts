/**
 * Auto-train scheduling service.
 *
 * Periodically checks the server for new unlabeled recordings. When enough
 * new data has accumulated since the last training run, automatically
 * triggers classifier training via POST /activity/train.
 *
 * Default configuration:
 *   - Check interval: 5 minutes (300000ms)
 *   - Min new recordings to trigger training: 10
 *   - Cooldown: don't re-train within 2 minutes of a previous train
 */

import AsyncStorage from '@react-native-async-storage/async-storage';
import {getApiUrl} from './api-client';

const CONFIG_KEY = '@sloughgpt/auto_train_config';
const LAST_TRAIN_KEY = '@sloughgpt/last_auto_train';
const LAST_COUNT_KEY = '@sloughgpt/last_recording_count';

export interface AutoTrainConfig {
  enabled: boolean;
  intervalMs: number;
  minNewRecordings: number;
  cooldownMs: number;
}

const DEFAULT_CONFIG: AutoTrainConfig = {
  enabled: true,
  intervalMs: 300_000,     // 5 min
  minNewRecordings: 10,
  cooldownMs: 120_000,     // 2 min
};

let _config: AutoTrainConfig = {...DEFAULT_CONFIG};
let _timer: ReturnType<typeof setInterval> | null = null;
let _running = false;

export async function getAutoTrainConfig(): Promise<AutoTrainConfig> {
  try {
    const raw = await AsyncStorage.getItem(CONFIG_KEY);
    if (raw) _config = {..._config, ...JSON.parse(raw)};
  } catch {}
  return _config;
}

export async function setAutoTrainConfig(partial: Partial<AutoTrainConfig>): Promise<void> {
  _config = {..._config, ...partial};
  await AsyncStorage.setItem(CONFIG_KEY, JSON.stringify(_config));
  if (_timer) {
    stopAutoTrainScheduler();
    startAutoTrainScheduler();
  }
}

async function getLastTrainTime(): Promise<number> {
  try {
    const raw = await AsyncStorage.getItem(LAST_TRAIN_KEY);
    return raw ? Number(raw) : 0;
  } catch {
    return 0;
  }
}

async function getLastRecordingCount(): Promise<number> {
  try {
    const raw = await AsyncStorage.getItem(LAST_COUNT_KEY);
    return raw ? Number(raw) : 0;
  } catch {
    return 0;
  }
}

async function checkAndTrain(): Promise<void> {
  if (_running) return;
  _running = true;
  try {
    const baseUrl = await getApiUrl();

    // Check recording count
    const res = await fetch(`${baseUrl}/activity/status`);
    if (!res.ok) return;
    const status = await res.json();
    const currentCount = status.num_recordings || 0;

    const lastCount = await getLastRecordingCount();
    const lastTrain = await getLastTrainTime();
    const now = Date.now();

    // Skip if server was just trained by another client
    if (currentCount <= lastCount) {
      await AsyncStorage.setItem(LAST_COUNT_KEY, String(currentCount));
      return;
    }

    const newRecordings = currentCount - lastCount;

    // Check cooldown
    if (now - lastTrain < _config.cooldownMs) return;

    if (newRecordings >= _config.minNewRecordings) {
      // Trigger training
      const trainRes = await fetch(`${baseUrl}/activity/train`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({epochs: 15, lr: 0.001, batch_size: 16}),
      });
      if (trainRes.ok) {
        await AsyncStorage.setItem(LAST_TRAIN_KEY, String(Date.now()));
      }
    }

    await AsyncStorage.setItem(LAST_COUNT_KEY, String(currentCount));
  } catch {
    // Server unreachable — try again next interval
  } finally {
    _running = false;
  }
}

export function startAutoTrainScheduler(): void {
  if (_timer) return;
  _timer = setInterval(checkAndTrain, _config.intervalMs);
  // Also run immediately
  checkAndTrain();
}

export function stopAutoTrainScheduler(): void {
  if (_timer) {
    clearInterval(_timer);
    _timer = null;
  }
  _running = false;
}

export function isAutoTrainRunning(): boolean {
  return _timer !== null;
}
