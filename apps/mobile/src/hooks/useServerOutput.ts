import {useEffect, useRef, useState, useCallback} from 'react';
import {streamSSE, type SSEEvent} from '../services/sse-client';

export interface OutputLine {
  id: string;
  message: string;
  severity: 'info' | 'warning' | 'error' | 'debug';
  timestamp: number;
}

interface UseServerOutputOptions {
  maxLines?: number;
  tail?: number;
  enabled?: boolean;
}

export function useServerOutput(options: UseServerOutputOptions = {}) {
  const {maxLines = 200, tail = 50, enabled = true} = options;
  const [lines, setLines] = useState<OutputLine[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [paused, setPaused] = useState(false);
  const pausedRef = useRef(false);
  const linesRef = useRef<OutputLine[]>([]);
  const abortRef = useRef<AbortController | null>(null);

  const addLine = useCallback((msg: string, severity: OutputLine['severity'] = 'info') => {
    if (pausedRef.current) return;
    const line: OutputLine = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
      message: msg,
      severity,
      timestamp: Date.now(),
    };
    linesRef.current = [...linesRef.current, line].slice(-maxLines);
    setLines([...linesRef.current]);
  }, [maxLines]);

  const clear = useCallback(() => {
    linesRef.current = [];
    setLines([]);
  }, []);

  const togglePause = useCallback(() => {
    pausedRef.current = !pausedRef.current;
    setPaused(pausedRef.current);
  }, []);

  const exportLines = useCallback(() => {
    return linesRef.current.map(l => {
      const ts = new Date(l.timestamp).toISOString();
      return `[${ts}] [${l.severity}] ${l.message}`;
    }).join('\n');
  }, []);

  useEffect(() => {
    if (!enabled) return;

    const startStream = async () => {
      setStreaming(true);
      abortRef.current = new AbortController();

      try {
        for await (const event of streamSSE('/server/output', {tail}, abortRef.current.signal)) {
          if (event.data?.line) {
            const line = String(event.data.line);
            const severity = (event.data.severity as OutputLine['severity']) || 'info';
            addLine(line, severity);
          }
          if (event.data?.message) {
            addLine(String(event.data.message), (event.data.severity as OutputLine['severity']) || 'info');
          }
        }
      } catch (err: any) {
        if (err.name !== 'AbortError' && err.name !== 'SSEHttpError') {
          addLine(`Stream error: ${err.message}`, 'error');
        }
      } finally {
        setStreaming(false);
      }
    };

    startStream();

    return () => {
      abortRef.current?.abort();
    };
  }, [enabled, tail, addLine]);

  return {
    lines,
    streaming,
    paused,
    clear,
    togglePause,
    exportLines,
  };
}
