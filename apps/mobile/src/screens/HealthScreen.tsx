import React, {useEffect, useState, useRef} from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  RefreshControl,
} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {api} from '../services/api-client';
import {StatusBadge} from '../components/StatusBadge';
import {colors, spacing, radii, typography} from '../theme';
import type {DetailedHealth} from '../types';

function ProgressBar({value, color}: {value: number; color: string}) {
  return (
    <View style={styles.progressTrack}>
      <View style={[styles.progressFill, {width: `${Math.min(value, 100)}%`, backgroundColor: color}]} />
    </View>
  );
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
    <SafeAreaView style={styles.safe} edges={['top']}>
      <ScrollView
        contentContainerStyle={styles.content}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }>
        <Text style={styles.title}>System Health</Text>

        <View style={styles.kpiRow}>
          <View style={styles.kpiCard}>
            <Text style={styles.kpiLabel}>API</Text>
            <StatusBadge
              label={api_?.status === 'healthy' ? 'Healthy' : 'Offline'}
              variant={api_?.status === 'healthy' ? 'success' : 'error'}
            />
          </View>
          <View style={styles.kpiCard}>
            <Text style={styles.kpiLabel}>Model</Text>
            <StatusBadge
              label={api_?.model_loaded ? 'Loaded' : 'None'}
              variant={api_?.model_loaded ? 'success' : 'default'}
            />
          </View>
        </View>

        {api_?.model_name && (
          <View style={styles.card}>
            <Text style={styles.cardTitle}>Active Model</Text>
            <Text style={styles.cardValue}>{api_.model_name}</Text>
          </View>
        )}

        {sys && (
          <>
            <View style={styles.card}>
              <Text style={styles.cardTitle}>CPU</Text>
              <Text style={styles.cardValue}>{sys.cpu_percent.toFixed(1)}%</Text>
              <ProgressBar value={sys.cpu_percent} color={colors.primary} />
            </View>

            <View style={styles.card}>
              <Text style={styles.cardTitle}>Memory</Text>
              <Text style={styles.cardValue}>
                {sys.memory_used_gb.toFixed(1)} / {sys.memory_total_gb.toFixed(1)} GB
                ({sys.memory_percent.toFixed(1)}%)
              </Text>
              <ProgressBar value={sys.memory_percent} color={colors.accent} />
            </View>

            {sys.disk_total_gb > 0 && (
              <View style={styles.card}>
                <Text style={styles.cardTitle}>Disk</Text>
                <Text style={styles.cardValue}>
                  {sys.disk_used_gb.toFixed(1)} / {sys.disk_total_gb.toFixed(1)} GB
                  ({((sys.disk_used_gb / sys.disk_total_gb) * 100).toFixed(1)}%)
                </Text>
                <ProgressBar
                  value={(sys.disk_used_gb / sys.disk_total_gb) * 100}
                  color={colors.warning}
                />
              </View>
            )}

            <View style={styles.card}>
              <Text style={styles.cardTitle}>Uptime</Text>
              <Text style={styles.cardValue}>
                {formatUptime(sys.uptime)}
              </Text>
            </View>
          </>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

function formatUptime(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  const h = Math.floor(seconds / 3600);
  const m = Math.round((seconds % 3600) / 60);
  return `${h}h ${m}m`;
}

const styles = StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: colors.background,
  },
  content: {
    padding: spacing.lg,
    gap: spacing.md,
  },
  title: {
    ...typography.h1,
    color: colors.text,
  },
  kpiRow: {
    flexDirection: 'row',
    gap: spacing.sm,
  },
  kpiCard: {
    flex: 1,
    backgroundColor: colors.surface,
    borderRadius: radii.lg,
    padding: spacing.md,
    alignItems: 'center',
    gap: spacing.sm,
  },
  kpiLabel: {
    ...typography.small,
    color: colors.textMuted,
    textTransform: 'uppercase',
  },
  card: {
    backgroundColor: colors.surface,
    borderRadius: radii.lg,
    padding: spacing.lg,
  },
  cardTitle: {
    ...typography.small,
    color: colors.textMuted,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    marginBottom: spacing.xs,
  },
  cardValue: {
    ...typography.h3,
    color: colors.text,
    marginBottom: spacing.sm,
  },
  progressTrack: {
    height: 6,
    backgroundColor: colors.border,
    borderRadius: 3,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    borderRadius: 3,
  },
});
