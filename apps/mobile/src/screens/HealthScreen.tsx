import React, {useEffect, useState, useRef, useCallback} from 'react';
import {FlatList, Pressable, RefreshControl} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {YStack, XStack, Text} from 'tamagui';
import {useColors} from '../theme/colors';
import {api} from '../services/api-client';
import {StatusBadge} from '../components/StatusBadge';
import {Icon} from '../components/Icon';

interface DetailedHealth {
  status: string;
  uptime_seconds: number;
  timestamp: string;
  request_count: number;
  error_count: number;
  avg_latency_ms: number;
  requests_per_minute: number;
  inference_count: number;
  total_tokens: number;
  tokens_per_sec: number;
  avg_tokens_per_request: number;
  model_loaded: boolean;
  model_loading: boolean;
  model_type: string | null;
  device: string | null;
  soul: string | null;
  system: {
    cpu_percent: number;
    memory_percent: number;
    memory_available_mb: number;
    open_files?: number;
    threads?: number;
    process_cpu_percent?: number;
    rss_mb?: number;
  };
  gpu: {
    backend: string;
    device_type?: string;
    vram_gb?: number;
    tier?: string;
    error?: string;
  };
  health_score: {
    score: number;
    status: string;
    summary?: string;
    diagnoses?: string[];
  };
  kv_sessions: {
    enabled: boolean;
    active_sessions: number;
    cached_tokens: number;
    ttl_seconds?: number;
  };
  training_pool: {
    active_jobs: number;
    max_workers: number;
    total_tracked: number;
  };
  lifecycle: {
    phase: string;
    is_running: boolean;
    in_flight: number;
  };
  recent_errors: Array<{timestamp: string; message: string; path?: string}>;
  status_message: string;
}

function Card({children, style}: {children: React.ReactNode; style?: any}) {
  const colors = useColors();
  return (
    <YStack
      padding={14}
      borderRadius={12}
      backgroundColor={colors.white}
      borderWidth={0.5}
      borderColor={colors.border}
      gap={8}
      {...style}>
      {children}
    </YStack>
  );
}

function Stat({label, value, color}: {label: string; value: string; color?: string}) {
  const colors = useColors();
  return (
    <YStack gap={2}>
      <Text fontSize={10} color={colors.textMuted} letterSpacing={0.3}>{label}</Text>
      <Text fontSize={14} fontWeight="600" color={color || colors.text}>{value}</Text>
    </YStack>
  );
}

function ProgressBar({value, color}: {value: number; color: string}) {
  const colors = useColors();
  return (
    <YStack height={5} backgroundColor={colors.border} borderRadius={3} overflow="hidden">
      <YStack height="100%" width={`${Math.min(value, 100)}%`} backgroundColor={color} borderRadius={3} />
    </YStack>
  );
}

function formatUptime(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  const h = Math.floor(seconds / 3600);
  const m = Math.round((seconds % 3600) / 60);
  if (h < 24) return `${h}h ${m}m`;
  const d = Math.floor(h / 24);
  return `${d}d ${h % 24}h`;
}

function HealthScoreBadge({score, status}: {score: number; status: string}) {
  const colors = useColors();
  const variant = score >= 80 ? 'success' : score >= 50 ? 'warning' : 'error';
  return (
    <XStack alignItems="center" gap={8}>
      <Text fontSize={28} fontWeight="700" color={variant === 'success' ? colors.success : variant === 'warning' ? colors.warning : colors.error}>
        {score}
      </Text>
      <YStack gap={2}>
        <Text fontSize={11} color={colors.textMuted}>/ 100</Text>
        <StatusBadge label={status} variant={variant} />
      </YStack>
    </XStack>
  );
}

export function HealthScreen() {
  const colors = useColors();
  const [health, setHealth] = useState<DetailedHealth | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchHealth = useCallback(async () => {
    try {
      const data = await api.get<DetailedHealth>('/health/detailed');
      setHealth(data);
    } catch {
      try {
        const basic = await api.get<any>('/health');
        setHealth({
          status: basic.status || 'unknown',
          uptime_seconds: basic.uptime || 0,
          timestamp: new Date().toISOString(),
          request_count: 0,
          error_count: 0,
          avg_latency_ms: 0,
          requests_per_minute: 0,
          inference_count: 0,
          total_tokens: 0,
          tokens_per_sec: 0,
          avg_tokens_per_request: 0,
          model_loaded: basic.model_loaded || false,
          model_loading: false,
          model_type: basic.model_name || null,
          device: null,
          soul: null,
          system: {cpu_percent: 0, memory_percent: 0, memory_available_mb: 0},
          gpu: {backend: 'unknown'},
          health_score: {score: 0, status: 'unknown'},
          kv_sessions: {enabled: false, active_sessions: 0, cached_tokens: 0},
          training_pool: {active_jobs: 0, max_workers: 0, total_tracked: 0},
          lifecycle: {phase: 'unknown', is_running: false, in_flight: 0},
          recent_errors: [],
          status_message: basic.status || 'Unknown',
        });
      } catch {}
    }
  }, []);

  useEffect(() => {
    fetchHealth().finally(() => setLoading(false));
    intervalRef.current = setInterval(fetchHealth, 5000);
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [fetchHealth]);

  const onRefresh = async () => {
    setRefreshing(true);
    await fetchHealth();
    setRefreshing(false);
  };

  if (loading) {
    return (
      <SafeAreaView style={{flex: 1, backgroundColor: colors.background}} edges={['top']}>
        <YStack flex={1} alignItems="center" justifyContent="center">
          <StatusBadge label="Loading..." variant="info" />
        </YStack>
      </SafeAreaView>
    );
  }

  const s = health?.system;
  const g = health?.gpu;
  const hs = health?.health_score;
  const kv = health?.kv_sessions;
  const tp = health?.training_pool;
  const lc = health?.lifecycle;

  const sections: React.ReactNode[] = [];

  // Health Score
  if (hs && hs.score > 0) {
    sections.push(
      <Card key="score">
        <XStack justifyContent="space-between" alignItems="center">
          <Text fontSize={13} fontWeight="600" color={colors.text}>Health Score</Text>
          <HealthScoreBadge score={hs.score} status={hs.status} />
        </XStack>
        {hs.diagnoses && hs.diagnoses.length > 0 && (
          <YStack gap={4} marginTop={4}>
            {hs.diagnoses.slice(0, 3).map((d, i) => (
              <XStack key={i} gap={6} alignItems="center">
                <Icon name="triangle-alert" size={12} color={colors.warning} />
                <Text fontSize={11} color={colors.textMuted} flex={1}>{d}</Text>
              </XStack>
            ))}
          </YStack>
        )}
      </Card>,
    );
  }

  // Status Row
  sections.push(
    <Card key="status">
      <XStack gap={8}>
        <YStack flex={1} alignItems="center" gap={4}>
          <Text fontSize={10} color={colors.textMuted} letterSpacing={0.3}>API</Text>
          <StatusBadge label={health?.status === 'healthy' ? 'Healthy' : health?.status || 'Unknown'} variant={health?.status === 'healthy' ? 'success' : 'error'} />
        </YStack>
        <YStack flex={1} alignItems="center" gap={4}>
          <Text fontSize={10} color={colors.textMuted} letterSpacing={0.3}>MODEL</Text>
          <StatusBadge label={health?.model_loaded ? 'Loaded' : health?.model_loading ? 'Loading' : 'None'} variant={health?.model_loaded ? 'success' : health?.model_loading ? 'warning' : 'default'} />
        </YStack>
        <YStack flex={1} alignItems="center" gap={4}>
          <Text fontSize={10} color={colors.textMuted} letterSpacing={0.3}>LIFECYCLE</Text>
          <StatusBadge label={lc?.phase || '—'} variant={lc?.is_running ? 'success' : 'default'} />
        </YStack>
      </XStack>
    </Card>,
  );

  // Model Info
  if (health?.model_type || health?.soul || health?.device) {
    sections.push(
      <Card key="model">
        <Text fontSize={13} fontWeight="600" color={colors.text}>Model</Text>
        <XStack gap={16} flexWrap="wrap">
          {health.model_type && <Stat label="Type" value={health.model_type} />}
          {health.soul && <Stat label="Soul" value={health.soul} color={colors.primary} />}
          {health.device && <Stat label="Device" value={health.device} />}
        </XStack>
      </Card>,
    );
  }

  // CPU + Memory
  if (s) {
    sections.push(
      <Card key="resources">
        <Text fontSize={13} fontWeight="600" color={colors.text}>Resources</Text>
        <YStack gap={10}>
          <YStack gap={4}>
            <XStack justifyContent="space-between">
              <Text fontSize={12} color={colors.textMuted}>CPU</Text>
              <Text fontSize={12} fontWeight="500" color={colors.text}>{s.cpu_percent.toFixed(1)}%</Text>
            </XStack>
            <ProgressBar value={s.cpu_percent} color={colors.primary} />
          </YStack>
          <YStack gap={4}>
            <XStack justifyContent="space-between">
              <Text fontSize={12} color={colors.textMuted}>Memory</Text>
              <Text fontSize={12} fontWeight="500" color={colors.text}>{s.memory_percent.toFixed(1)}%</Text>
            </XStack>
            <ProgressBar value={s.memory_percent} color={colors.warning} />
          </YStack>
          <XStack gap={16}>
            {s.open_files != null && <Stat label="Open files" value={String(s.open_files)} />}
            {s.threads != null && <Stat label="Threads" value={String(s.threads)} />}
            {s.rss_mb != null && <Stat label="RSS" value={`${(s.rss_mb / 1024).toFixed(1)} GB`} />}
          </XStack>
        </YStack>
      </Card>,
    );
  }

  // GPU
  if (g && g.backend !== 'unknown') {
    sections.push(
      <Card key="gpu">
        <XStack justifyContent="space-between" alignItems="center">
          <Text fontSize={13} fontWeight="600" color={colors.text}>GPU</Text>
          <StatusBadge label={g.tier || g.backend} variant={g.tier ? 'success' : 'default'} />
        </XStack>
        <XStack gap={16}>
          <Stat label="Backend" value={g.backend} />
          {g.vram_gb != null && <Stat label="VRAM" value={`${g.vram_gb} GB`} />}
          {g.device_type && <Stat label="Device" value={g.device_type} />}
        </XStack>
      </Card>,
    );
  }

  // Inference
  if (health && health.inference_count > 0) {
    sections.push(
      <Card key="inference">
        <Text fontSize={13} fontWeight="600" color={colors.text}>Inference</Text>
        <XStack gap={16} flexWrap="wrap">
          <Stat label="Requests" value={String(health.inference_count)} />
          <Stat label="Tokens/sec" value={health.tokens_per_sec.toFixed(1)} />
          <Stat label="Total tokens" value={health.total_tokens.toLocaleString()} />
          {health.avg_latency_ms > 0 && <Stat label="Avg latency" value={`${health.avg_latency_ms.toFixed(0)}ms`} />}
        </XStack>
      </Card>,
    );
  }

  // KV Cache
  if (kv && kv.enabled) {
    sections.push(
      <Card key="kv">
        <XStack justifyContent="space-between" alignItems="center">
          <Text fontSize={13} fontWeight="600" color={colors.text}>KV Cache</Text>
          <StatusBadge label={kv.active_sessions > 0 ? `${kv.active_sessions} sessions` : 'Idle'} variant={kv.active_sessions > 0 ? 'info' : 'default'} />
        </XStack>
        <XStack gap={16}>
          <Stat label="Cached tokens" value={kv.cached_tokens.toLocaleString()} />
          {kv.ttl_seconds != null && <Stat label="TTL" value={`${kv.ttl_seconds}s`} />}
        </XStack>
      </Card>,
    );
  }

  // Training Pool
  if (tp && tp.max_workers > 0) {
    sections.push(
      <Card key="training">
        <Text fontSize={13} fontWeight="600" color={colors.text}>Training Pool</Text>
        <XStack gap={16}>
          <Stat label="Active" value={`${tp.active_jobs} / ${tp.max_workers}`} color={tp.active_jobs > 0 ? colors.warning : colors.text} />
          <Stat label="Tracked" value={String(tp.total_tracked)} />
        </XStack>
      </Card>,
    );
  }

  // Recent Errors
  if (health?.recent_errors && health.recent_errors.length > 0) {
    sections.push(
      <Card key="errors">
        <XStack justifyContent="space-between" alignItems="center">
          <Text fontSize={13} fontWeight="600" color={colors.text}>Recent Errors</Text>
          <StatusBadge label={`${health.recent_errors.length}`} variant="error" />
        </XStack>
        <YStack gap={6}>
          {health.recent_errors.slice(0, 3).map((err, i) => (
            <YStack key={i} padding={8} borderRadius={6} backgroundColor={colors.errorAlpha(0.05)} gap={2}>
              <Text fontSize={11} color={colors.error} numberOfLines={1}>{err.message}</Text>
              {err.path && <Text fontSize={10} color={colors.textMuted}>{err.path}</Text>}
            </YStack>
          ))}
        </YStack>
      </Card>,
    );
  }

  // Uptime
  sections.push(
    <Card key="uptime">
      <XStack justifyContent="space-between" alignItems="center">
        <Text fontSize={13} fontWeight="600" color={colors.text}>Uptime</Text>
        <Text fontSize={14} fontWeight="500" color={colors.text}>{formatUptime(health?.uptime_seconds || 0)}</Text>
      </XStack>
      <XStack gap={16}>
        <Stat label="Requests" value={String(health?.request_count || 0)} />
        <Stat label="Errors" value={String(health?.error_count || 0)} color={health?.error_count ? colors.error : colors.text} />
        {health && health.requests_per_minute > 0 && <Stat label="Req/min" value={health.requests_per_minute.toFixed(1)} />}
      </XStack>
    </Card>,
  );

  return (
    <SafeAreaView style={{flex: 1, backgroundColor: colors.background}} edges={['top']}>
      <XStack paddingHorizontal={16} paddingVertical={12} alignItems="center" justifyContent="space-between">
        <Text fontSize={20} fontWeight="600" color={colors.text}>System Health</Text>
        <Pressable onPress={onRefresh}>
          <Icon name="refresh-cw" size={18} color={colors.primary} />
        </Pressable>
      </XStack>
      <FlatList
        data={sections}
        renderItem={({item}) => item as React.ReactElement}
        keyExtractor={(_, i) => String(i)}
        contentContainerStyle={{padding: 16, gap: 12, paddingBottom: 32}}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}
      />
    </SafeAreaView>
  );
}
