/**
 * Connection status hook — polls server health, tracks connection state.
 * Returns detailed status: 'connected' | 'connecting' | 'reconnecting' | 'offline'.
 * Includes model state, latency, and last-seen timestamp.
 */

import {useEffect, useState, useRef, useCallback} from 'react';
import {getApiUrl} from '../services/api-client';

export type ConnectionState = 'connected' | 'connecting' | 'reconnecting' | 'offline';

interface ConnectionInfo {
  state: ConnectionState;
  latencyMs: number | null;
  modelLoaded: boolean;
  lastSeen: number | null;
  retryCount: number;
}

const POLL_INTERVAL = 8000;

export function useConnectionStatus(): ConnectionInfo {
  const [info, setInfo] = useState<ConnectionInfo>({
    state: 'connecting',
    latencyMs: null,
    modelLoaded: false,
    lastSeen: null,
    retryCount: 0,
  });
  const mountedRef = useRef(true);

  const check = useCallback(async () => {
    if (!mountedRef.current) return;

    try {
      const url = await getApiUrl();
      const start = Date.now();
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);
      const res = await fetch(`${url}/health`, {method: 'GET', signal: controller.signal});
      clearTimeout(timeoutId);
      const latency = Date.now() - start;

      if (res.ok) {
        const data = await res.json();
        if (!mountedRef.current) return;

        setInfo({
          state: 'connected',
          latencyMs: latency,
          modelLoaded: data.model_loaded ?? false,
          lastSeen: Date.now(),
          retryCount: 0,
        });
      } else {
        if (!mountedRef.current) return;
        setInfo(prev => ({
          ...prev,
          state: 'reconnecting',
          retryCount: prev.retryCount + 1,
        }));
      }
    } catch {
      if (!mountedRef.current) return;
      setInfo(prev => ({
        ...prev,
        state: 'offline',
        latencyMs: null,
        retryCount: prev.retryCount + 1,
      }));
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    check();
    const interval = setInterval(check, POLL_INTERVAL);
    return () => {
      mountedRef.current = false;
      clearInterval(interval);
    };
  }, [check]);

  return info;
}
