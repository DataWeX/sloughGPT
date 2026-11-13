import {useEffect, useRef, useState, useCallback} from 'react';
import {streamSSE, type SSEEvent} from '../services/sse-client';
import {api} from '../services/api-client';

export type ConnectionStatus = 'connected' | 'disconnected' | 'connecting';

export interface LiveHealth {
  status: string;
  model_loaded: boolean;
  model_loading: boolean;
  model_type: string | null;
  soul: string | null;
  uptime: number;
  inference_count: number;
  total_tokens: number;
  tokens_per_sec: number;
  avg_latency_ms: number;
  cpu_percent: number;
  memory_percent: number;
  memory_used_gb: number;
  memory_total_gb: number;
  request_count: number;
  error_count: number;
  requests_per_minute: number;
  health_score: number;
  status_message: string;
}

interface UseLiveStatusOptions {
  pollIntervalMs?: number;
  enabled?: boolean;
}

export function useLiveStatus(options: UseLiveStatusOptions = {}) {
  const {pollIntervalMs = 5000, enabled = true} = options;
  const [health, setHealth] = useState<LiveHealth | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('connecting');
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const mountedRef = useRef(true);

  const fetchHealth = useCallback(async () => {
    if (!enabled || !mountedRef.current) return;
    try {
      const data = await api.get<LiveHealth>('/health');
      if (!mountedRef.current) return;
      setHealth(data);
      setConnectionStatus('connected');
    } catch {
      if (!mountedRef.current) return;
      setConnectionStatus('disconnected');
    }
  }, [enabled]);

  useEffect(() => {
    mountedRef.current = true;
    fetchHealth();
    intervalRef.current = setInterval(fetchHealth, pollIntervalMs);
    return () => {
      mountedRef.current = false;
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [fetchHealth, pollIntervalMs]);

  return {health, connectionStatus, refresh: fetchHealth};
}
