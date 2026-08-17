import React, {useEffect, useState, useRef} from 'react';
import {YStack, XStack, Text} from 'tamagui';
import {ScreenShell} from '../components/ScreenShell';
import {api} from '../services/api-client';
import {StatusBadge} from '../components/StatusBadge';
import type {DetailedHealth} from '../types';

function ProgressBar({value, color}: {value: number; color: string}) {
  return (
    <YStack height={6} backgroundColor="$borderColor" borderRadius={3} overflow="hidden">
      <YStack height="100%" width={`${Math.min(value, 100)}%`} backgroundColor={color} borderRadius={3} />
    </YStack>
  );
}

function formatUptime(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  const h = Math.floor(seconds / 3600);
  const m = Math.round((seconds % 3600) / 60);
  return `${h}h ${m}m`;
}

export function HealthScreen() {
  const [health, setHealth] = useState<DetailedHealth | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchHealth = async () => {
    try {
      const data = await api.get<DetailedHealth>('/health/detailed');
      setHealth(data);
    } catch {
      try {
        const basic = await api.get<any>('/health');
        setHealth({
          api: {status: basic.status, model_loaded: basic.model_loaded, model_name: basic.model_name},
          system: {
            cpu_percent: 0,
            memory_percent: 0,
            memory_used_gb: 0,
            memory_total_gb: 0,
            disk_used_gb: 0,
            disk_free_gb: 0,
            disk_total_gb: 0,
            uptime: basic.uptime || 0,
          },
        });
      } catch {}
    }
  };

  useEffect(() => {
    fetchHealth();
    intervalRef.current = setInterval(fetchHealth, 5000);
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);

  const onRefresh = async () => {
    setRefreshing(true);
    await fetchHealth();
    setRefreshing(false);
  };

  const sys = health?.system;
  const api_ = health?.api;

  return (
    <ScreenShell title="System Health" refreshing={refreshing} onRefresh={onRefresh}>
      <XStack gap={12}>
        <YStack flex={1} backgroundColor="$background" borderRadius={12} padding={14} alignItems="center" gap={8} borderWidth={0.5} borderColor="$borderColor">
          <Text fontSize={11} fontWeight="500" letterSpacing={0.5} color="$color10" textTransform="uppercase">
            API
          </Text>
          <StatusBadge label={api_?.status === 'healthy' ? 'Healthy' : 'Offline'} variant={api_?.status === 'healthy' ? 'success' : 'error'} />
        </YStack>
        <YStack flex={1} backgroundColor="$background" borderRadius={12} padding={14} alignItems="center" gap={8} borderWidth={0.5} borderColor="$borderColor">
          <Text fontSize={11} fontWeight="500" letterSpacing={0.5} color="$color10" textTransform="uppercase">
            Model
          </Text>
          <StatusBadge label={api_?.model_loaded ? 'Loaded' : 'None'} variant={api_?.model_loaded ? 'success' : 'default'} />
        </YStack>
      </XStack>

      {api_?.model_name && (
        <YStack backgroundColor="$background" borderRadius={12} padding={16} borderWidth={0.5} borderColor="$borderColor">
          <Text fontSize={11} fontWeight="500" letterSpacing={0.5} color="$color10" textTransform="uppercase" marginBottom={4}>
            Active Model
          </Text>
          <Text fontSize={16} fontWeight="600" color="$color" marginBottom={8}>
            {api_.model_name}
          </Text>
        </YStack>
      )}

      {sys && (
        <>
          <YStack backgroundColor="$background" borderRadius={12} padding={16} borderWidth={0.5} borderColor="$borderColor">
            <Text fontSize={11} fontWeight="500" letterSpacing={0.5} color="$color10" textTransform="uppercase" marginBottom={4}>
              CPU
            </Text>
            <Text fontSize={16} fontWeight="600" color="$color" marginBottom={8}>
              {sys.cpu_percent.toFixed(1)}%
            </Text>
            <ProgressBar value={sys.cpu_percent} color="#7C52C4" />
          </YStack>

          <YStack backgroundColor="$background" borderRadius={12} padding={16} borderWidth={0.5} borderColor="$borderColor">
            <Text fontSize={11} fontWeight="500" letterSpacing={0.5} color="$color10" textTransform="uppercase" marginBottom={4}>
              Memory
            </Text>
            <Text fontSize={16} fontWeight="600" color="$color" marginBottom={8}>
              {sys.memory_used_gb.toFixed(1)} / {sys.memory_total_gb.toFixed(1)} GB ({sys.memory_percent.toFixed(1)}%)
            </Text>
            <ProgressBar value={sys.memory_percent} color="#F0935C" />
          </YStack>

          {sys.disk_total_gb > 0 && (
            <YStack backgroundColor="$background" borderRadius={12} padding={16} borderWidth={0.5} borderColor="$borderColor">
              <Text fontSize={11} fontWeight="500" letterSpacing={0.5} color="$color10" textTransform="uppercase" marginBottom={4}>
                Disk
              </Text>
              <Text fontSize={16} fontWeight="600" color="$color" marginBottom={8}>
                {sys.disk_used_gb.toFixed(1)} / {sys.disk_total_gb.toFixed(1)} GB ({((sys.disk_used_gb / sys.disk_total_gb) * 100).toFixed(1)}%)
              </Text>
              <ProgressBar value={(sys.disk_used_gb / (sys.disk_used_gb + sys.disk_free_gb)) * 100} color="#E8A83C" />
            </YStack>
          )}

          <YStack backgroundColor="$background" borderRadius={12} padding={16} borderWidth={0.5} borderColor="$borderColor">
            <Text fontSize={11} fontWeight="500" letterSpacing={0.5} color="$color10" textTransform="uppercase" marginBottom={4}>
              Uptime
            </Text>
            <Text fontSize={16} fontWeight="600" color="$color" marginBottom={8}>
              {formatUptime(sys.uptime)}
            </Text>
          </YStack>
        </>
      )}

      {health?.inference && (
        <YStack backgroundColor="$background" borderRadius={12} padding={16} borderWidth={0.5} borderColor="$borderColor">
          <Text fontSize={11} fontWeight="500" letterSpacing={0.5} color="$color10" textTransform="uppercase" marginBottom={8}>
            Inference
          </Text>
          <XStack gap={16}>
            <YStack flex={1}>
              <Text fontSize={12} color="$color10">Requests</Text>
              <Text fontSize={16} fontWeight="600" color="$color">{health.inference.inference_count}</Text>
            </YStack>
            <YStack flex={1}>
              <Text fontSize={12} color="$color10">Tokens/sec</Text>
              <Text fontSize={16} fontWeight="600" color="$color">{health.inference.avg_tokens_per_sec.toFixed(1)}</Text>
            </YStack>
            <YStack flex={1}>
              <Text fontSize={12} color="$color10">Total tokens</Text>
              <Text fontSize={16} fontWeight="600" color="$color">{health.inference.total_tokens.toLocaleString()}</Text>
            </YStack>
          </XStack>
        </YStack>
      )}

      {health?.services && (
        <YStack backgroundColor="$background" borderRadius={12} padding={16} borderWidth={0.5} borderColor="$borderColor">
          <Text fontSize={11} fontWeight="500" letterSpacing={0.5} color="$color10" textTransform="uppercase" marginBottom={8}>
            Services
          </Text>
          <YStack gap={10}>
            {health.services.training_pool && (
              <XStack justifyContent="space-between" alignItems="center">
                <Text fontSize={14} color="$color">Training Pool</Text>
                <StatusBadge
                  label={`${health.services.training_pool.active}/${health.services.training_pool.max} active`}
                  variant={health.services.training_pool.active > 0 ? 'warning' : 'success'}
                />
              </XStack>
            )}
            {health.services.inference_pool && (
              <XStack justifyContent="space-between" alignItems="center">
                <Text fontSize={14} color="$color">Inference Pool</Text>
                <StatusBadge
                  label={`${health.services.inference_pool.workers} workers`}
                  variant={health.services.inference_pool.active > 0 ? 'warning' : 'success'}
                />
              </XStack>
            )}
          </YStack>
        </YStack>
      )}
    </ScreenShell>
  );
}
