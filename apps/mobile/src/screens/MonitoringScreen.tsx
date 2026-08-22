import React, {useEffect, useState, useCallback, useRef} from 'react';
import {FlatList, Pressable, RefreshControl, ScrollView} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {YStack, XStack, Text} from 'tamagui';
import {useColors} from '../theme/colors';
import {api} from '../services/api-client';
import {StatusBadge} from '../components/StatusBadge';
import {Icon} from '../components/Icon';
import {useLiveStatus, type LiveHealth, type ConnectionStatus} from '../hooks/useLiveStatus';

interface DetailedHealth {
  status: string;
  uptime_seconds: number;
  timestamp: string;
  request_count: number;
  error_count: number;
  avg_latency_ms: number;
  p95_latency_ms: number;
  requests_per_minute: number;
  path_latencies: Array<{path: string; avg_ms: number; count: number; p95_ms: number}>;
  recent_errors: Array<{path: string; method: string; status: number; message: string; error_type: string; ts: number}>;
  inference_count: number;
  total_tokens: number;
  tokens_per_sec: number;
  avg_tokens_per_request: number;
  health_score: {score: number; status: string};
  status_message: string;
  model_metrics: Array<{model: string; count: number; total_tokens: number; tokens_per_sec: number; avg_tokens: number}>;
  model_events: Array<{type: string; model: string; detail: string; ts: number}>;
  rate_violations: Array<{path: string; count: number; limit: number; ts: number}>;
  system: {
    cpu_percent: number;
    memory_percent: number;
    memory_available_mb: number;
    open_files?: number;
    threads?: number;
    process_cpu_percent?: number;
    rss_mb?: number;
  };
  gpu: {backend: string; device_type?: string; vram_gb?: number; tier?: string};
  kv_sessions: {enabled: boolean; active_sessions: number; cached_tokens: number; ttl_seconds?: number};
  training_pool: {active_jobs: number; max_workers: number; total_tracked: number};
}

interface SystemMetrics {
  cpu_percent: number;
  memory_percent: number;
  memory_used_gb: number;
  memory_total_gb: number;
}

interface DiskUsage {
  total_gb: number;
  used_gb: number;
  free_gb: number;
  percent: number;
}

interface TrainingJob {
  id: string;
  status: string;
  method: string;
  started_at: string;
  progress?: number;
  loss?: number;
}

const POLL_INTERVAL = 5000;
const MAX_HISTORY = 30;

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
    <YStack gap={2} flex={1}>
      <Text fontSize={10} color={colors.textMuted} letterSpacing={0.3}>{label}</Text>
      <Text fontSize={14} fontWeight="600" color={color || colors.text} numberOfLines={1}>{value}</Text>
    </YStack>
  );
}

function ProgressBar({value, color, max = 100}: {value: number; color: string; max?: number}) {
  const colors = useColors();
  const pct = Math.min((value / max) * 100, 100);
  return (
    <YStack height={5} backgroundColor={colors.border} borderRadius={3} overflow="hidden">
      <YStack height="100%" width={`${pct}%`} backgroundColor={color} borderRadius={3} />
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

function MiniChart({data, color, height = 40}: {data: number[]; color: string; height?: number}) {
  const colors = useColors();
  if (data.length < 2) return null;
  const max = Math.max(...data, 1);
  const width = 200;
  const step = width / (data.length - 1);

  return (
    <YStack height={height} width="100%" position="relative">
      {data.map((val, i) => {
        const x = i * step;
        const y = height - (val / max) * (height - 4);
        return (
          <YStack
            key={i}
            position="absolute"
            left={x}
            bottom={y}
            width={3}
            height={3}
            borderRadius={1.5}
            backgroundColor={color}
            opacity={0.3 + (i / data.length) * 0.7}
          />
        );
      })}
    </YStack>
  );
}

export function MonitoringScreen() {
  const colors = useColors();
  const {health: liveHealth, connectionStatus, refresh: refreshLive} = useLiveStatus({pollIntervalMs: POLL_INTERVAL});
  const [detailed, setDetailed] = useState<DetailedHealth | null>(null);
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null);
  const [disk, setDisk] = useState<DiskUsage | null>(null);
  const [trainingJobs, setTrainingJobs] = useState<TrainingJob[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [lastUpdated, setLastUpdated] = useState<string | null>(null);
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set(['status', 'resources']));
  const [cpuHistory, setCpuHistory] = useState<number[]>([]);
  const [memHistory, setMemHistory] = useState<number[]>([]);

  const toggleSection = useCallback((key: string) => {
    setExpandedSections(prev => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key); else next.add(key);
      return next;
    });
  }, []);

  const fetchAll = useCallback(async (showRefreshing = false) => {
    if (showRefreshing) setRefreshing(true);
    try {
      const [d, m, di, tj] = await Promise.all([
        api.get<DetailedHealth>('/health/detailed').catch(() => null),
        api.get<SystemMetrics>('/system/metrics').catch(() => null),
        api.get<DiskUsage>('/system/disk').catch(() => null),
        api.get<TrainingJob[]>('/training/jobs').catch(() => []),
      ]);
      if (d) setDetailed(d);
      if (m) setMetrics(m);
      if (di) setDisk(di);
      if (Array.isArray(tj)) setTrainingJobs(tj);
      setLastUpdated(new Date().toLocaleTimeString());
    } catch {}
    if (showRefreshing) setRefreshing(false);
  }, []);

  useEffect(() => {
    fetchAll().finally(() => setLoading(false));
  }, [fetchAll]);

  useEffect(() => {
    if (!autoRefresh || connectionStatus !== 'connected') return;
    const timer = setInterval(fetchAll, POLL_INTERVAL);
    return () => clearInterval(timer);
  }, [autoRefresh, connectionStatus, fetchAll]);

  useEffect(() => {
    if (!liveHealth) return;
    setCpuHistory(prev => {
      const next = [...prev, liveHealth.cpu_percent];
      return next.length > MAX_HISTORY ? next.slice(-MAX_HISTORY) : next;
    });
    setMemHistory(prev => {
      const next = [...prev, liveHealth.memory_percent];
      return next.length > MAX_HISTORY ? next.slice(-MAX_HISTORY) : next;
    });
  }, [liveHealth]);

  const sections: {key: string; title: string; content: React.ReactNode}[] = [];

  // Status Overview
  sections.push({
    key: 'status',
    title: 'Status',
    content: (
      <Card>
        <XStack gap={8}>
          <YStack flex={1} alignItems="center" gap={4}>
            <Text fontSize={10} color={colors.textMuted}>API</Text>
            <StatusBadge
              label={liveHealth?.status === 'healthy' ? 'Healthy' : liveHealth?.status || 'Unknown'}
              variant={liveHealth?.status === 'healthy' ? 'success' : 'error'}
            />
          </YStack>
          <YStack flex={1} alignItems="center" gap={4}>
            <Text fontSize={10} color={colors.textMuted}>MODEL</Text>
            <StatusBadge
              label={liveHealth?.model_loaded ? 'Loaded' : liveHealth?.model_loading ? 'Loading' : 'None'}
              variant={liveHealth?.model_loaded ? 'success' : liveHealth?.model_loading ? 'warning' : 'default'}
            />
          </YStack>
          <YStack flex={1} alignItems="center" gap={4}>
            <Text fontSize={10} color={colors.textMuted}>CONNECTION</Text>
            <StatusBadge
              label={connectionStatus}
              variant={connectionStatus === 'connected' ? 'success' : connectionStatus === 'connecting' ? 'warning' : 'error'}
            />
          </YStack>
        </XStack>
        {liveHealth?.model_type && (
          <XStack gap={16} marginTop={4}>
            <Stat label="Model" value={liveHealth.model_type} color={colors.primary} />
            {liveHealth.soul && <Stat label="Soul" value={liveHealth.soul} color={colors.accent} />}
          </XStack>
        )}
        {liveHealth && (
          <XStack gap={16}>
            <Stat label="Uptime" value={formatUptime(liveHealth.uptime)} />
            <Stat label="Requests" value={String(liveHealth.request_count)} />
            <Stat label="Errors" value={String(liveHealth.error_count)} color={liveHealth.error_count > 0 ? colors.error : colors.text} />
          </XStack>
        )}
      </Card>
    ),
  });

  // Resources
  sections.push({
    key: 'resources',
    title: 'Resources',
    content: (
      <Card>
        <Text fontSize={13} fontWeight="600" color={colors.text}>Resources</Text>
        <YStack gap={10}>
          <YStack gap={4}>
            <XStack justifyContent="space-between">
              <Text fontSize={12} color={colors.textMuted}>CPU</Text>
              <Text fontSize={12} fontWeight="500" color={colors.text}>{(liveHealth?.cpu_percent ?? 0).toFixed(1)}%</Text>
            </XStack>
            <ProgressBar value={liveHealth?.cpu_percent ?? 0} color={colors.primary} />
            <MiniChart data={cpuHistory} color={colors.primary} />
          </YStack>
          <YStack gap={4}>
            <XStack justifyContent="space-between">
              <Text fontSize={12} color={colors.textMuted}>Memory</Text>
              <Text fontSize={12} fontWeight="500" color={colors.text}>{(liveHealth?.memory_percent ?? 0).toFixed(1)}%</Text>
            </XStack>
            <ProgressBar value={liveHealth?.memory_percent ?? 0} color={colors.warning} />
            <MiniChart data={memHistory} color={colors.warning} />
          </YStack>
          {metrics && (
            <XStack gap={16}>
              <Stat label="Used" value={`${metrics.memory_used_gb.toFixed(1)} GB`} />
              <Stat label="Total" value={`${metrics.memory_total_gb.toFixed(1)} GB`} />
            </XStack>
          )}
          {detailed?.system && (
            <XStack gap={16}>
              {detailed.system.open_files != null && <Stat label="Files" value={String(detailed.system.open_files)} />}
              {detailed.system.threads != null && <Stat label="Threads" value={String(detailed.system.threads)} />}
              {detailed.system.rss_mb != null && <Stat label="RSS" value={`${(detailed.system.rss_mb / 1024).toFixed(1)} GB`} />}
            </XStack>
          )}
        </YStack>
      </Card>
    ),
  });

  // Inference Metrics
  if (liveHealth && liveHealth.inference_count > 0) {
    sections.push({
      key: 'inference',
      title: 'Inference',
      content: (
        <Card>
          <Text fontSize={13} fontWeight="600" color={colors.text}>Inference</Text>
          <XStack gap={16} flexWrap="wrap">
            <Stat label="Requests" value={String(liveHealth.inference_count)} />
            <Stat label="Tokens/sec" value={liveHealth.tokens_per_sec.toFixed(1)} color={colors.success} />
            <Stat label="Total tokens" value={liveHealth.total_tokens.toLocaleString()} />
            <Stat label="Avg latency" value={`${liveHealth.avg_latency_ms.toFixed(0)}ms`} />
          </XStack>
        </Card>
      ),
    });
  }

  // Model Metrics
  if (detailed?.model_metrics && detailed.model_metrics.length > 0) {
    sections.push({
      key: 'model-metrics',
      title: 'Model Metrics',
      content: (
        <Card>
          <Text fontSize={13} fontWeight="600" color={colors.text}>Model Metrics</Text>
          <YStack gap={8}>
            {detailed.model_metrics.map((m, i) => (
              <YStack key={i} backgroundColor={colors.muted} padding={10} borderRadius={8} gap={4}>
                <XStack justifyContent="space-between">
                  <Text fontSize={12} fontWeight="500" color={colors.text} numberOfLines={1}>{m.model}</Text>
                  <Text fontSize={11} color={colors.textMuted}>{m.count} reqs</Text>
                </XStack>
                <XStack gap={16}>
                  <Stat label="Tokens/s" value={m.tokens_per_sec.toFixed(1)} />
                  <Stat label="Total" value={m.total_tokens.toLocaleString()} />
                  <Stat label="Avg" value={m.avg_tokens.toFixed(0)} />
                </XStack>
              </YStack>
            ))}
          </YStack>
        </Card>
      ),
    });
  }

  // GPU
  if (detailed?.gpu && detailed.gpu.backend !== 'unknown') {
    sections.push({
      key: 'gpu',
      title: 'GPU',
      content: (
        <Card>
          <XStack justifyContent="space-between" alignItems="center">
            <Text fontSize={13} fontWeight="600" color={colors.text}>GPU</Text>
            <StatusBadge label={detailed.gpu.tier || detailed.gpu.backend} variant={detailed.gpu.tier ? 'success' : 'default'} />
          </XStack>
          <XStack gap={16}>
            <Stat label="Backend" value={detailed.gpu.backend} />
            {detailed.gpu.vram_gb != null && <Stat label="VRAM" value={`${detailed.gpu.vram_gb} GB`} />}
            {detailed.gpu.device_type && <Stat label="Device" value={detailed.gpu.device_type} />}
          </XStack>
        </Card>
      ),
    });
  }

  // Disk
  if (disk) {
    sections.push({
      key: 'disk',
      title: 'Disk',
      content: (
        <Card>
          <Text fontSize={13} fontWeight="600" color={colors.text}>Disk</Text>
          <YStack gap={4}>
            <XStack justifyContent="space-between">
              <Text fontSize={12} color={colors.textMuted}>Usage</Text>
              <Text fontSize={12} fontWeight="500" color={colors.text}>{disk.percent.toFixed(1)}%</Text>
            </XStack>
            <ProgressBar value={disk.percent} color={colors.info} />
            <XStack gap={16}>
              <Stat label="Used" value={`${disk.used_gb.toFixed(1)} GB`} />
              <Stat label="Free" value={`${disk.free_gb.toFixed(1)} GB`} />
              <Stat label="Total" value={`${disk.total_gb.toFixed(1)} GB`} />
            </XStack>
          </YStack>
        </Card>
      ),
    });
  }

  // KV Cache
  if (detailed?.kv_sessions?.enabled) {
    sections.push({
      key: 'kv-cache',
      title: 'KV Cache',
      content: (
        <Card>
          <XStack justifyContent="space-between" alignItems="center">
            <Text fontSize={13} fontWeight="600" color={colors.text}>KV Cache</Text>
            <StatusBadge
              label={detailed.kv_sessions.active_sessions > 0 ? `${detailed.kv_sessions.active_sessions} sessions` : 'Idle'}
              variant={detailed.kv_sessions.active_sessions > 0 ? 'info' : 'default'}
            />
          </XStack>
          <XStack gap={16}>
            <Stat label="Cached tokens" value={detailed.kv_sessions.cached_tokens.toLocaleString()} />
            {detailed.kv_sessions.ttl_seconds != null && <Stat label="TTL" value={`${detailed.kv_sessions.ttl_seconds}s`} />}
          </XStack>
        </Card>
      ),
    });
  }

  // Training Pool
  if (detailed?.training_pool && detailed.training_pool.max_workers > 0) {
    sections.push({
      key: 'training-pool',
      title: 'Training Pool',
      content: (
        <Card>
          <Text fontSize={13} fontWeight="600" color={colors.text}>Training Pool</Text>
          <XStack gap={16}>
            <Stat label="Active" value={`${detailed.training_pool.active_jobs} / ${detailed.training_pool.max_workers}`} color={detailed.training_pool.active_jobs > 0 ? colors.warning : colors.text} />
            <Stat label="Tracked" value={String(detailed.training_pool.total_tracked)} />
          </XStack>
        </Card>
      ),
    });
  }

  // Path Latencies
  if (detailed?.path_latencies && detailed.path_latencies.length > 0) {
    sections.push({
      key: 'latencies',
      title: 'Path Latencies',
      content: (
        <Card>
          <Text fontSize={13} fontWeight="600" color={colors.text}>Path Latencies</Text>
          <YStack gap={6}>
            {detailed.path_latencies.slice(0, 8).map((p, i) => (
              <XStack key={i} justifyContent="space-between" alignItems="center">
                <Text fontSize={11} color={colors.textMuted} numberOfLines={1} flex={1}>{p.path}</Text>
                <XStack gap={12}>
                  <Text fontSize={11} color={colors.text}>{p.avg_ms.toFixed(0)}ms</Text>
                  <Text fontSize={10} color={colors.textMuted}>p95: {p.p95_ms.toFixed(0)}ms</Text>
                </XStack>
              </XStack>
            ))}
          </YStack>
        </Card>
      ),
    });
  }

  // Recent Errors
  if (detailed?.recent_errors && detailed.recent_errors.length > 0) {
    sections.push({
      key: 'errors',
      title: 'Recent Errors',
      content: (
        <Card>
          <XStack justifyContent="space-between" alignItems="center">
            <Text fontSize={13} fontWeight="600" color={colors.text}>Recent Errors</Text>
            <StatusBadge label={`${detailed.recent_errors.length}`} variant="error" />
          </XStack>
          <YStack gap={6}>
            {detailed.recent_errors.slice(0, 5).map((err, i) => (
              <YStack key={i} padding={8} borderRadius={6} backgroundColor={colors.errorAlpha(0.05)} gap={2}>
                <Text fontSize={11} color={colors.error} numberOfLines={1}>{err.message}</Text>
                <XStack gap={8}>
                  <Text fontSize={10} color={colors.textMuted}>{err.path}</Text>
                  <Text fontSize={10} color={colors.textMuted}>{err.status}</Text>
                </XStack>
              </YStack>
            ))}
          </YStack>
        </Card>
      ),
    });
  }

  // Model Events
  if (detailed?.model_events && detailed.model_events.length > 0) {
    sections.push({
      key: 'model-events',
      title: 'Model Events',
      content: (
        <Card>
          <Text fontSize={13} fontWeight="600" color={colors.text}>Model Events</Text>
          <YStack gap={6}>
            {detailed.model_events.slice(0, 5).map((evt, i) => (
              <YStack key={i} padding={8} borderRadius={6} backgroundColor={colors.muted} gap={2}>
                <XStack gap={8} alignItems="center">
                  <StatusBadge label={evt.type} variant={evt.type === 'load' ? 'success' : evt.type === 'unload' ? 'warning' : 'default'} />
                  <Text fontSize={11} fontWeight="500" color={colors.text} numberOfLines={1}>{evt.model}</Text>
                </XStack>
                <Text fontSize={10} color={colors.textMuted}>{evt.detail}</Text>
              </YStack>
            ))}
          </YStack>
        </Card>
      ),
    });
  }

  // Training Jobs
  if (trainingJobs.length > 0) {
    sections.push({
      key: 'training-jobs',
      title: 'Training Jobs',
      content: (
        <Card>
          <Text fontSize={13} fontWeight="600" color={colors.text}>Training Jobs</Text>
          <YStack gap={6}>
            {trainingJobs.slice(0, 5).map((job) => (
              <YStack key={job.id} padding={10} borderRadius={8} backgroundColor={colors.muted} gap={4}>
                <XStack justifyContent="space-between" alignItems="center">
                  <Text fontSize={12} fontWeight="500" color={colors.text} numberOfLines={1}>{job.method}</Text>
                  <StatusBadge
                    label={job.status}
                    variant={job.status === 'completed' ? 'success' : job.status === 'running' ? 'info' : job.status === 'failed' ? 'error' : 'default'}
                  />
                </XStack>
                {job.progress != null && (
                  <ProgressBar value={job.progress * 100} color={colors.primary} />
                )}
                {job.loss != null && (
                  <Text fontSize={10} color={colors.textMuted}>Loss: {job.loss.toFixed(4)}</Text>
                )}
                <Text fontSize={10} color={colors.textMuted}>Started: {new Date(job.started_at).toLocaleString()}</Text>
              </YStack>
            ))}
          </YStack>
        </Card>
      ),
    });
  }

  const headerRight = (
    <XStack alignItems="center" gap={12}>
      {lastUpdated && (
        <Text fontSize={10} color={colors.textMuted}>Updated {lastUpdated}</Text>
      )}
      <Pressable onPress={() => setAutoRefresh(!autoRefresh)}>
        <XStack alignItems="center" gap={4} padding={6} borderRadius={6} backgroundColor={autoRefresh ? colors.successAlpha(0.1) : colors.muted}>
          <Icon name={autoRefresh ? 'pause' : 'play'} size={12} color={autoRefresh ? colors.success : colors.textMuted} />
          <Text fontSize={10} color={autoRefresh ? colors.success : colors.textMuted}>{autoRefresh ? 'Auto' : 'Manual'}</Text>
        </XStack>
      </Pressable>
      <Pressable onPress={() => fetchAll(true)} disabled={refreshing}>
        <Icon name="refresh-cw" size={18} color={colors.primary} />
      </Pressable>
    </XStack>
  );

  return (
    <SafeAreaView style={{flex: 1, backgroundColor: colors.background}} edges={['top']}>
      <YStack paddingHorizontal={16} paddingVertical={12}>
        <XStack justifyContent="space-between" alignItems="center">
          <YStack>
            <Text fontSize={20} fontWeight="600" color={colors.text}>System Health</Text>
            <Text fontSize={12} color={colors.textMuted}>Real-time monitoring</Text>
          </YStack>
          {headerRight}
        </XStack>
      </YStack>

      {loading ? (
        <YStack flex={1} alignItems="center" justifyContent="center">
          <StatusBadge label="Loading..." variant="info" />
        </YStack>
      ) : (
        <FlatList
          data={sections}
          renderItem={({item}) => (
            <YStack paddingHorizontal={16} paddingBottom={12}>
              <Pressable onPress={() => toggleSection(item.key)}>
                <XStack justifyContent="space-between" alignItems="center" paddingVertical={8}>
                  <Text fontSize={13} fontWeight="600" color={colors.text}>{item.title}</Text>
                  <Icon name={expandedSections.has(item.key) ? 'chevron-up' : 'chevron-down'} size={16} color={colors.textMuted} />
                </XStack>
              </Pressable>
              {expandedSections.has(item.key) && item.content}
            </YStack>
          )}
          keyExtractor={item => item.key}
          contentContainerStyle={{paddingBottom: 32}}
          refreshControl={<RefreshControl refreshing={refreshing} onRefresh={() => fetchAll(true)} />}
        />
      )}
    </SafeAreaView>
  );
}
