/**
 * Background sensor recording service.
 *
 * Collects accelerometer + gyroscope data when the app is in the background
 * (or suspended). Buffers readings in AsyncStorage, periodically flushes
 * them to the server as labeled/unlabeled recordings.
 *
 * Requires:
 *   npm install react-native-sensors
 *   cd ios && pod install
 *
 * Without `react-native-sensors`, falls back to simulated data for development.
 * The mock data simulates a "walking" pattern so the classifier can be tested
 * end-to-end.
 */

import AsyncStorage from '@react-native-async-storage/async-storage';
import {getApiUrl} from './api-client';
import type {SensorReading} from '../types';

const BUFFER_KEY = '@sloughgpt/sensor_buffer';
const RECORDING_KEY = '@sloughgpt/recording_active';
const LAST_SYNC_KEY = '@sloughgpt/last_sensor_sync';

const WINDOW_SIZE = 128;       // samples per recording window
const SYNC_INTERVAL_MS = 60_000;  // flush to server every 60s

interface BufferedReadings {
  readings: SensorReading[];
  startTime: number;
}

async function getBuffer(): Promise<BufferedReadings> {
  try {
    const raw = await AsyncStorage.getItem(BUFFER_KEY);
    if (raw) return JSON.parse(raw);
  } catch {}
  return {readings: [], startTime: Date.now()};
}

async function setBuffer(buf: BufferedReadings): Promise<void> {
  await AsyncStorage.setItem(BUFFER_KEY, JSON.stringify(buf));
}

async function flushBuffer(): Promise<void> {
  const buf = await getBuffer();
  if (buf.readings.length < 10) return;

  const baseUrl = await getApiUrl();
  const chunks: SensorReading[][] = [];
  for (let i = 0; i < buf.readings.length; i += WINDOW_SIZE) {
    chunks.push(buf.readings.slice(i, i + WINDOW_SIZE));
  }

  let uploaded = 0;
  for (const chunk of chunks) {
    if (chunk.length < 10) continue;
    const data = chunk.map(r => [
      r.accel.x, r.accel.y, r.accel.z,
      r.gyro.x, r.gyro.y, r.gyro.z,
    ]);
    try {
      const res = await fetch(`${baseUrl}/activity/data`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({data, label: null}),
      });
      if (res.ok) uploaded++;
    } catch {
      // Server unreachable — keep buffer for next flush
      return;
    }
  }

  if (uploaded > 0) {
    await AsyncStorage.setItem(LAST_SYNC_KEY, String(Date.now()));
    // Clear only the flushed portion
    const remaining = buf.readings.slice(chunks.length * WINDOW_SIZE);
    await setBuffer({readings: remaining, startTime: remaining.length > 0 ? buf.startTime : Date.now()});
  }
}

// ── Simulated background recording ──────────────────────────────────────
let _mockInterval: ReturnType<typeof setInterval> | null = null;
let _mockPhase = 0;

function startMockRecording(onReading: (r: SensorReading) => void): () => void {
  const interval = setInterval(() => {
    const t = (_mockPhase * 100) / 1000;
    onReading({
      timestamp: Date.now(),
      accel: {
        x: Math.sin(t * 2) + (Math.random() - 0.5) * 0.1,
        y: Math.cos(t * 1.7) + (Math.random() - 0.5) * 0.1,
        z: 9.81 + Math.sin(t * 0.5) + (Math.random() - 0.5) * 0.1,
      },
      gyro: {
        x: Math.sin(t * 3.1) * 0.8 + (Math.random() - 0.5) * 0.05,
        y: Math.cos(t * 2.3) * 0.5 + (Math.random() - 0.5) * 0.05,
        z: Math.sin(t * 1.1) * 0.3 + (Math.random() - 0.5) * 0.05,
      },
    });
    _mockPhase++;
  }, 100);
  _mockInterval = interval;
  return () => {
    clearInterval(interval);
    _mockInterval = null;
  };
}

// ── Public API ──────────────────────────────────────────────────────────

let _cleanup: (() => void) | null = null;
let _flushTimer: ReturnType<typeof setInterval> | null = null;

export interface BackgroundRecorderState {
  active: boolean;
  bufferSize: number;
  lastSync: number | null;
}

export function getBackgroundRecorderState(): BackgroundRecorderState {
  return {
    active: _cleanup !== null,
    bufferSize: 0,  // populated async
    lastSync: null,
  };
}

/** Start background recording. Returns cleanup function. */
export async function startBackgroundRecording(): Promise<() => void> {
  const already = await AsyncStorage.getItem(RECORDING_KEY);
  if (already === 'true') {
    // Already recording — return noop cleanup
    return () => {};
  }

  await AsyncStorage.setItem(RECORDING_KEY, 'true');

  const onReading = async (r: SensorReading) => {
    const buf = await getBuffer();
    buf.readings.push(r);
    await setBuffer(buf);
  };

  // Try real sensor; fall back to mock
  let cleanup: (() => void) | null = null;
  try {
    const {accelerometer, gyroscope} = require('react-native-sensors');
    const {combineLatest} = require('rxjs');
    const accel$ = accelerometer({updateInterval: 100, enable: true});
    const gyro$ = gyroscope({updateInterval: 100, enable: true});
    const sub = combineLatest([accel$, gyro$]).subscribe({
      next: ([a, g]: any) => {
        onReading({
          timestamp: Date.now(),
          accel: {x: a.x, y: a.y, z: a.z},
          gyro: {x: g.x, y: g.y, z: g.z},
        });
      },
    });
    cleanup = () => sub.unsubscribe();
  } catch {
    cleanup = startMockRecording(r => { onReading(r); });
  }

  _cleanup = cleanup;

  // Periodic flush
  _flushTimer = setInterval(() => { flushBuffer(); }, SYNC_INTERVAL_MS);

  return () => stopBackgroundRecording();
}

/** Stop background recording and flush remaining buffer. */
export async function stopBackgroundRecording(): Promise<void> {
  _cleanup?.();
  _cleanup = null;
  if (_flushTimer) {
    clearInterval(_flushTimer);
    _flushTimer = null;
  }
  await AsyncStorage.setItem(RECORDING_KEY, 'false');
  await flushBuffer();
}

/** Get current buffer size (number of unsynced readings). */
export async function getBufferSize(): Promise<number> {
  const buf = await getBuffer();
  return buf.readings.length;
}

/** Get timestamp of last successful sync. */
export async function getLastSyncTime(): Promise<number | null> {
  try {
    const raw = await AsyncStorage.getItem(LAST_SYNC_KEY);
    return raw ? Number(raw) : null;
  } catch {
    return null;
  }
}
