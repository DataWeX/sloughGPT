import { useState, useEffect, useCallback } from 'react';
import {
  SloughGPTClient,
  SloughGPTConfig,
  HealthStatus,
  GenerateRequest,
  ChatRequest,
  ChatMessage,
} from './client';

export interface UseSloughGPTOptions extends SloughGPTConfig {
  autoConnect?: boolean;
}

export function useSloughGPT(options: UseSloughGPTOptions = {}) {
  const [client] = useState(() => new SloughGPTClient(options));
  const [isReady, setIsReady] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [health, setHealth] = useState<HealthStatus | null>(null);

  const checkHealth = useCallback(async () => {
    try {
      const status = await client.health();
      setHealth(status);
      setIsReady(status.status === 'healthy');
      setError(null);
    } catch (e) {
      setError(e as Error);
      setIsReady(false);
    }
  }, [client]);

  useEffect(() => {
    if (options.autoConnect !== false) {
      checkHealth();
    }
  }, [checkHealth, options.autoConnect]);

  const generate = useCallback(
    async (prompt: string, opts?: Partial<GenerateRequest>): Promise<string> => {
      setIsLoading(true);
      try {
        const result = await client.generate({ prompt, ...opts });
        return result.text;
      } finally {
        setIsLoading(false);
      }
    },
    [client]
  );

  const chat = useCallback(
    async (
      messages: ChatMessage[],
      opts?: Partial<ChatRequest>
    ): Promise<string> => {
      setIsLoading(true);
      try {
        const result = await client.chat({ messages, ...opts });
        return result.message.content;
      } finally {
        setIsLoading(false);
      }
    },
    [client]
  );

  return {
    client,
    isReady,
    isLoading,
    error,
    health,
    generate,
    chat,
    checkHealth,
  };
}
