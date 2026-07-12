/**
 * Background sensor collection for activity classification.
 *
 * Uses react-native-sensors (accelerometer + gyroscope) to collect
 * 6-axis motion data at 50 Hz.  Data is batched and sent to the
 * server's activity classifier training endpoint.
 *
 * Data flow:
 *   sensor readings → sliding window buffer → POST /activity/record
 *   → server trains CNN classifier → predictions returned
 *
 * Usage:
 *   import {startCollection, stopCollection} from './sensor-collection';
 *   startCollection({activity: 'walking', intervalMs: 50});
 *   // ... later
 *   stopCollection();
 */

import {Dimensions} from 'react-native';
import {api} from './api-client';

// ── Types ─────────────────────────────────────────────────────────────

export interface SensorReading {
  timestamp: number;
  accelerometer: {x: number; y: number; z: number};
  gyroscope: {x: number; y: number; z: number};
}

export interface CollectionConfig {
  /** Activity label for this collection session */
  activity: string;
  /** Sensor sampling interval in ms (default: 50 = 20 Hz) */
  intervalMs?: number;
  /** Window size in readings before flushing to server */
  windowSize?: number;
}

export interface CollectionStats {
  totalReadings: number;
  totalWindows: number;
  lastFlushTime: number | null;
  isCollecting: boolean;
}

// ── State ─────────────────────────────────────────────────────────────

let _subscription: any = null;
let _buffer: SensorReading[] = [];
let _config: CollectionConfig | null = null;
let _stats: CollectionStats = {
  totalReadings: 0,
  totalWindows: 0,
  lastFlushTime: null,
  isCollecting: false,
};

// ── Sensor setup ──────────────────────────────────────────────────────

function createSensorObserver() {
  try {
    const {
      accelerometer,
      gyroscope,
      setUpdateIntervalForType,
      SensorTypes,
    } = require('react-native-sensors');

    // Set sampling interval
    const interval = _config?.intervalMs ?? 50;
    setUpdateIntervalForType(SensorTypes.accelerometer, interval);
    setUpdateIntervalForType(SensorTypes.gyroscope, interval);

    let accelData: {x: number; y: number; z: number} | null = null;
    let gyroData: {x: number; y: number; z: number} | null = null;

    const accelSub = accelerometer.subscribe(
      ({x, y, z}: {x: number; y: number; z: number}) => {
        accelData = {x, y, z};
        tryFlush();
      },
      (error: any) => console.warn('Accelerometer error:', error),
    );

    const gyroSub = gyroscope.subscribe(
      ({x, y, z}: {x: number; y: number; z: number}) => {
        gyroData = {x, y, z};
        tryFlush();
      },
      (error: any) => console.warn('Gyroscope error:', error),
    );

    return {
      unsubscribe: () => {
        accelSub.unsubscribe();
        gyroSub.unsubscribe();
      },
      getLatest: () => ({accel: accelData, gyro: gyroData}),
    };
  } catch (e) {
    console.warn('react-native-sensors not available:', e);
    return null;
  }
}

let _observer: ReturnType<typeof createSensorObserver> | null = null;

// ── Buffer & flush ────────────────────────────────────────────────────

function tryFlush() {
  if (!_observer || !_config) return;

  const {accel, gyro} = _observer.getLatest();
  if (!accel || !gyro) return;

  const reading: SensorReading = {
    timestamp: Date.now(),
    accelerometer: accel,
    gyroscope: gyro,
  };

  _buffer.push(reading);
  _stats.totalReadings++;

  const windowSize = _config.windowSize ?? 50;
  if (_buffer.length >= windowSize) {
    flushBuffer();
  }
}

async function flushBuffer() {
  if (_buffer.length === 0 || !_config) return;

  const readings = [..._buffer];
  _buffer = [];

  try {
    await api.post('/activity/record', {
      activity: _config.activity,
      readings: readings.map(r => [
        r.accelerometer.x, r.accelerometer.y, r.accelerometer.z,
        r.gyroscope.x, r.gyroscope.y, r.gyroscope.z,
      ]),
      timestamps: readings.map(r => r.timestamp),
    });
    _stats.totalWindows++;
    _stats.lastFlushTime = Date.now();
  } catch (e) {
    console.warn('Failed to flush sensor data:', e);
    // Re-add readings to buffer on failure
    _buffer = [...readings, ..._buffer];
  }
}

// ── Public API ────────────────────────────────────────────────────────

/**
 * Start collecting sensor data in the background.
 *
 * @param config - Collection configuration (activity label, interval, window)
 */
export function startCollection(config: CollectionConfig): void {
  if (_stats.isCollecting) {
    console.warn('Sensor collection already active');
    return;
  }

  _config = config;
  _buffer = [];
  _stats = {
    totalReadings: 0,
    totalWindows: 0,
    lastFlushTime: null,
    isCollecting: true,
  };

  _observer = createSensorObserver();
  if (!_observer) {
    _stats.isCollecting = false;
    console.warn('Could not start sensor collection');
    return;
  }

  console.log(
    `Sensor collection started: activity=${config.activity}, interval=${config.intervalMs ?? 50}ms`,
  );
}

/**
 * Stop collecting sensor data and flush remaining buffer.
 */
export async function stopCollection(): Promise<CollectionStats> {
  if (_observer) {
    _observer.unsubscribe();
    _observer = null;
  }

  // Flush any remaining data
  await flushBuffer();

  const stats = {..._stats, isCollecting: false};
  _stats.isCollecting = false;
  _config = null;

  console.log(
    `Sensor collection stopped: ${stats.totalReadings} readings, ${stats.totalWindows} windows`,
  );

  return stats;
}

/**
 * Get current collection statistics.
 */
export function getCollectionStats(): CollectionStats {
  return {..._stats};
}

/**
 * Check if sensor collection is active.
 */
export function isCollecting(): boolean {
  return _stats.isCollecting;
}
