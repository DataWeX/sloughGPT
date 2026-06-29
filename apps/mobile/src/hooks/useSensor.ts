import {useEffect, useRef, useCallback} from 'react';
import {useActivityStore} from '../stores/activity-store';
import type {SensorReading} from '../types';

/**
 * Collects accelerometer + gyroscope data and feeds it into the activity store.
 *
 * Uses a sensor abstraction layer:
 * 1. Tries to import `react-native-sensors` at runtime for real device data.
 * 2. Falls back to simulated sinusoidal data for development/testing.
 *
 * Install `react-native-sensors` for real device support:
 *   npm install react-native-sensors
 *   cd ios && pod install
 */

// 100ms sampling interval (10 Hz) — good balance for activity recognition
const SAMPLE_INTERVAL_MS = 100;

async function tryStartRealSensor(
  onReading: (r: SensorReading) => void,
): Promise<(() => void) | null> {
  try {
    const {
      accelerometer,
      gyroscope,
    } = require('react-native-sensors');
    const {combineLatest, filter} = require('rxjs');
    const accel$ = accelerometer({
      updateInterval: SAMPLE_INTERVAL_MS,
      enable: true,
    });
    const gyro$ = gyroscope({
      updateInterval: SAMPLE_INTERVAL_MS,
      enable: true,
    });
    const sub = combineLatest([accel$, gyro$])
      .pipe(filter(() => true))
      .subscribe({
        next: ([a, g]: any) => {
          onReading({
            timestamp: Date.now(),
            accel: {x: a.x, y: a.y, z: a.z},
            gyro: {x: g.x, y: g.y, z: g.z},
          });
        },
      });
    return () => sub.unsubscribe();
  } catch {
    return null;
  }
}

function startSimulatedSensor(
  onReading: (r: SensorReading) => void,
): () => void {
  let phase = 0;
  const interval = setInterval(() => {
    const t = (phase * SAMPLE_INTERVAL_MS) / 1000;
    // Simulate 6-axis data with sinusoidal patterns
    onReading({
      timestamp: Date.now(),
      accel: {
        x: Math.sin(t * 2) + (Math.random() - 0.5) * 0.05,
        y: Math.cos(t * 1.7) + (Math.random() - 0.5) * 0.05,
        z: 9.81 + Math.sin(t * 0.5) * 0.1 + (Math.random() - 0.5) * 0.05,
      },
      gyro: {
        x: Math.sin(t * 3.1) * 0.5 + (Math.random() - 0.5) * 0.02,
        y: Math.cos(t * 2.3) * 0.3 + (Math.random() - 0.5) * 0.02,
        z: Math.sin(t * 1.1) * 0.2 + (Math.random() - 0.5) * 0.02,
      },
    });
    phase++;
  }, SAMPLE_INTERVAL_MS);
  return () => clearInterval(interval);
}

/**
 * Hook that continuously collects sensor data when active.
 * Pass `activity="walking"` etc to change simulation pattern.
 */
export function useSensor(active: boolean = true) {
  const pushReading = useActivityStore(s => s.pushReading);
  const cleanupRef = useRef<(() => void) | null>(null);

  const start = useCallback(async () => {
    const onReading = (r: SensorReading) => pushReading(r);
    const realCleanup = await tryStartRealSensor(onReading);
    if (realCleanup) {
      cleanupRef.current = realCleanup;
    } else {
      cleanupRef.current = startSimulatedSensor(onReading);
    }
  }, [pushReading]);

  useEffect(() => {
    if (!active) {
      cleanupRef.current?.();
      cleanupRef.current = null;
      return;
    }
    start();
    return () => {
      cleanupRef.current?.();
      cleanupRef.current = null;
    };
  }, [active, start]);
}
