import React, {useEffect, useState, useCallback} from 'react';
import {FlatList, Pressable, RefreshControl} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {YStack, XStack, Text} from 'tamagui';
import {useColors} from '../theme/colors';
import {api} from '../services/api-client';
import {Icon} from '../components/Icon';
import {StatusBadge} from '../components/StatusBadge';
import {triggerHaptic} from '../services/haptics';
import {toast} from '../services/toast';
import type {BenchmarkResult} from '../types';

export function BenchmarkScreen() {
  const colors = useColors();
  const [results, setResults] = useState<BenchmarkResult[]>([]);
  const [running, setRunning] = useState(false);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const fetchResults = useCallback(async () => {
    try {
      const data = await api.get<{results: BenchmarkResult[]}>('/benchmark/metrics');
      setResults(data.results || []);
    } catch {
      setResults([]);
    }
  }, []);

  useEffect(() => {
    fetchResults().finally(() => setLoading(false));
  }, [fetchResults]);

  const onRefresh = async () => {
    setRefreshing(true);
    await fetchResults();
    setRefreshing(false);
  };

  const handleRunBenchmark = async () => {
    try {
      setRunning(true);
      triggerHaptic('light');
      const result = await api.post<{status: string}>('/benchmark/run', {});
      triggerHaptic('success');
      toast.success('Benchmark complete');
      await fetchResults();
    } catch {
      toast.error('Benchmark failed');
    } finally {
      setRunning(false);
    }
  };

  const formatScore = (v: number | null) => v != null ? v.toFixed(2) : '—';

  const renderItem = ({item}: {item: BenchmarkResult}) => (
    <YStack
      padding={12}
      borderRadius={10}
      borderWidth={0.5}
      borderColor={colors.border}
      backgroundColor={colors.white}
      gap={8}>
      <XStack justifyContent="space-between" alignItems="center">
        <Text fontSize={14} fontWeight="500" color={colors.text}>{item.model || item.model_id}</Text>
        <Text fontSize={11} color={colors.textMuted}>{item.timestamp ? new Date(item.timestamp).toLocaleDateString() : '—'}</Text>
      </XStack>
      <XStack gap={8} flexWrap="wrap">
        <YStack alignItems="center" gap={2}>
          <Text fontSize={11} color={colors.textMuted}>Coherence</Text>
          <Text fontSize={14} fontWeight="600" color={colors.text}>{formatScore(item.coherence)}</Text>
        </YStack>
        <YStack alignItems="center" gap={2}>
          <Text fontSize={11} color={colors.textMuted}>Repetition</Text>
          <Text fontSize={14} fontWeight="600" color={item.repetition > 0.5 ? colors.error : colors.text}>{formatScore(item.repetition)}</Text>
        </YStack>
        <YStack alignItems="center" gap={2}>
          <Text fontSize={11} color={colors.textMuted}>Avg Length</Text>
          <Text fontSize={14} fontWeight="600" color={colors.text}>{item.avg_length?.toFixed(0) ?? '—'}</Text>
        </YStack>
        <YStack alignItems="center" gap={2}>
          <Text fontSize={11} color={colors.textMuted}>Perplexity</Text>
          <Text fontSize={14} fontWeight="600" color={colors.text}>{formatScore(item.perplexity)}</Text>
        </YStack>
      </XStack>
    </YStack>
  );

  return (
    <SafeAreaView style={{flex: 1, backgroundColor: colors.background}} edges={['top']}>
      <XStack paddingHorizontal={16} paddingVertical={12} alignItems="center" justifyContent="space-between">
        <Text fontSize={20} fontWeight="600" color={colors.text}>Benchmarks</Text>
        <Pressable onPress={handleRunBenchmark} disabled={running}>
          <XStack paddingHorizontal={10} paddingVertical={5} borderRadius={6} backgroundColor={running ? colors.border : colors.primary} gap={4} alignItems="center">
            <Icon name={running ? 'refresh-cw' : 'zap'} size={14} color="white" />
            <Text fontSize={12} fontWeight="500" color="white">{running ? 'Running...' : 'Run'}</Text>
          </XStack>
        </Pressable>
      </XStack>

      {loading ? (
        <YStack flex={1} alignItems="center" justifyContent="center">
          <StatusBadge label="Loading..." variant="info" />
        </YStack>
      ) : results.length === 0 ? (
        <YStack flex={1} alignItems="center" justifyContent="center" gap={8}>
          <Icon name="bar-chart" size={32} color={colors.textMuted} />
          <Text fontSize={14} color={colors.textMuted}>No benchmark results</Text>
          <Text fontSize={12} color={colors.textMuted}>Run a benchmark to see metrics</Text>
        </YStack>
      ) : (
        <FlatList
          data={results}
          keyExtractor={(_, i) => String(i)}
          renderItem={renderItem}
          contentContainerStyle={{padding: 16, gap: 8}}
          refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}
        />
      )}
    </SafeAreaView>
  );
}
