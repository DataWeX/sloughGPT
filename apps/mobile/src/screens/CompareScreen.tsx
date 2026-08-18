import React, {useState, useEffect, useCallback} from 'react';
import {RefreshControl, Pressable, ScrollView} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {YStack, XStack, Text} from 'tamagui';
import {useColors} from '../theme/colors';
import {useModelStore} from '../stores/model-store';
import {api} from '../services/api-client';
import {StatusBadge} from '../components/StatusBadge';
import {Icon} from '../components/Icon';
import {toast} from '../services/toast';
import type {ModelInfo, BenchmarkResult} from '../types';

function formatSize(mb: number): string {
  if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`;
  return `${mb.toFixed(0)} MB`;
}

function formatNumber(n: number | null | undefined): string {
  if (n == null) return '—';
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return n.toFixed(0);
}

export function CompareScreen() {
  const colors = useColors();

  const {models, health, refresh} = useModelStore();
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [benchmarks, setBenchmarks] = useState<Record<string, BenchmarkResult>>({});
  const [selected, setSelected] = useState<Set<string>>(new Set());

  const loadData = useCallback(async () => {
    await refresh();
    // Fetch benchmarks for all models
    const results: Record<string, BenchmarkResult> = {};
    for (const m of models) {
      try {
        const b = await api.get<BenchmarkResult>(`/benchmark/${m.id}`);
        results[m.id] = b;
      } catch {
        // no benchmark data for this model
      }
    }
    setBenchmarks(results);
  }, [models, refresh]);

  useEffect(() => {
    loadData().finally(() => setLoading(false));
  }, [loadData]);

  const onRefresh = async () => {
    setRefreshing(true);
    await loadData();
    setRefreshing(false);
  };

  const toggleSelect = (id: string) => {
    setSelected(prev => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        if (next.size >= 4) {
          toast.info('Maximum 4 models to compare');
          return prev;
        }
        next.add(id);
      }
      return next;
    });
  };

  const selectedModels = models.filter(m => selected.has(m.id));
  const loadedModel = health?.model_name;

  if (loading) {
    return (
      <SafeAreaView style={{flex: 1}} edges={['top']}>
        <YStack flex={1} backgroundColor={colors.background} alignItems="center" justifyContent="center">
          <StatusBadge label="Loading..." variant="info" />
        </YStack>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={{flex: 1}} edges={['top']}>
      <ScrollView
        style={{flex: 1, backgroundColor: colors.background}}
        contentContainerStyle={{padding: 16, gap: 12}}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}>
        <XStack justifyContent="space-between" alignItems="center" paddingBottom={4}>
          <Text fontSize={24} fontWeight="700" letterSpacing={-0.3} color="$color">
            Compare
          </Text>
          {selected.size >= 2 && (
            <StatusBadge label={`${selected.size} selected`} variant="info" />
          )}
        </XStack>

        {/* Model selector */}
        <YStack
          backgroundColor={colors.white}
          borderRadius={12}
          borderWidth={0.5}
          borderColor={colors.border}
          padding={14}
          gap={10}>
          <XStack alignItems="center" gap={6}>
            <Icon name="package" size={16} color={colors.primary} />
            <Text fontSize={15} fontWeight="600" color="$color">Select Models</Text>
          </XStack>
          <Text fontSize={12} color="$color10">Tap to select 2–4 models for comparison</Text>

          {models.length === 0 ? (
            <YStack padding={20} alignItems="center">
              <Text fontSize={13} color="$color11">No models available</Text>
            </YStack>
          ) : (
            models.map(m => {
              const isSelected = selected.has(m.id);
              const isLoaded = m.id === loadedModel || m.name === loadedModel;
              return (
                <Pressable key={m.id} onPress={() => toggleSelect(m.id)}>
                  {({pressed}) => (
                    <XStack
                      padding={10}
                      borderRadius={8}
                      borderWidth={0.5}
                      borderColor={isSelected ? colors.primary : colors.border}
                      backgroundColor={isSelected ? colors.primary + '08' : pressed ? colors.primaryAlpha(0.04) : 'transparent'}
                      gap={10}
                      alignItems="center">
                      <YStack
                        width={20}
                        height={20}
                        borderRadius={4}
                        borderWidth={1}
                        borderColor={isSelected ? colors.primary : colors.border}
                        backgroundColor={isSelected ? colors.primary : 'transparent'}
                        alignItems="center"
                        justifyContent="center">
                        {isSelected && <Icon name="check" size={12} color={colors.white} />}
                      </YStack>
                      <YStack flex={1}>
                        <XStack alignItems="center" gap={6}>
                          <Text fontSize={13} fontWeight="500" color="$color" numberOfLines={1}>
                            {m.name || m.id}
                          </Text>
                          {isLoaded && (
                            <YStack
                              paddingHorizontal={5}
                              paddingVertical={1}
                              borderRadius={4}
                              backgroundColor={colors.successAlpha(0.1)}>
                              <Text fontSize={9} fontWeight="600" color={colors.success}>LOADED</Text>
                            </YStack>
                          )}
                        </XStack>
                        {m.description && (
                          <Text fontSize={11} color="$color10" numberOfLines={1} marginTop={1}>
                            {m.description}
                          </Text>
                        )}
                      </YStack>
                      {m.size_mb != null && (
                        <Text fontSize={11} color="$color10">{formatSize(m.size_mb)}</Text>
                      )}
                    </XStack>
                  )}
                </Pressable>
              );
            })
          )}
        </YStack>

        {/* Comparison table */}
        {selectedModels.length >= 2 && (
          <YStack
            backgroundColor={colors.white}
            borderRadius={12}
            borderWidth={0.5}
            borderColor={colors.border}
            padding={14}
            gap={10}>
            <XStack alignItems="center" gap={6}>
              <Icon name="bar-chart" size={16} color={colors.primary} />
              <Text fontSize={15} fontWeight="600" color="$color">Comparison</Text>
            </XStack>

            {/* Header row */}
            <XStack gap={0}>
              <YStack width={100} />
              {selectedModels.map(m => (
                <YStack key={m.id} flex={1} alignItems="center" paddingHorizontal={4}>
                  <Text fontSize={11} fontWeight="600" color="$color" numberOfLines={1} textAlign="center">
                    {(m.name || m.id).slice(0, 12)}
                  </Text>
                </YStack>
              ))}
            </XStack>

            {/* Rows */}
            {[
              {
                label: 'Parameters',
                getValue: (m: ModelInfo) => formatNumber(parseFloat(m.params) || 0),
              },
              {
                label: 'Size',
                getValue: (m: ModelInfo) => m.size_mb != null ? formatSize(m.size_mb) : '—',
              },
              {
                label: 'Source',
                getValue: (m: ModelInfo) => m.source || '—',
              },
              {
                label: 'Coherence',
                getValue: (m: ModelInfo) => {
                  const b = benchmarks[m.id];
                  return b ? `${(b.coherence * 100).toFixed(0)}%` : '—';
                },
              },
              {
                label: 'Repetition',
                getValue: (m: ModelInfo) => {
                  const b = benchmarks[m.id];
                  return b ? `${(b.repetition * 100).toFixed(0)}%` : '—';
                },
              },
              {
                label: 'Avg Length',
                getValue: (m: ModelInfo) => {
                  const b = benchmarks[m.id];
                  return b ? `${Number(b.avg_response_length ?? 0).toFixed(0)}` : '—';
                },
              },
            ].map((row, i) => (
              <XStack
                key={row.label}
                gap={0}
                paddingVertical={8}
                borderTopWidth={0.5}
                borderTopColor={colors.border}>
                <YStack width={100} justifyContent="center">
                  <Text fontSize={12} fontWeight="500" color="$color11">{row.label}</Text>
                </YStack>
                {selectedModels.map(m => (
                  <YStack key={m.id} flex={1} alignItems="center" paddingHorizontal={4} justifyContent="center">
                    <Text fontSize={12} color="$color">{row.getValue(m)}</Text>
                  </YStack>
                ))}
              </XStack>
            ))}
          </YStack>
        )}

        {selected.size < 2 && (
          <YStack
            backgroundColor={colors.white}
            borderRadius={12}
            borderWidth={0.5}
            borderColor={colors.border}
            padding={32}
            alignItems="center"
            gap={8}>
            <Icon name="package" size={28} color={colors.textMuted} />
            <Text fontSize={14} color="$color11" textAlign="center">
              Select 2 or more models to compare
            </Text>
            <Text fontSize={12} color="$color10" textAlign="center">
              Tap models above to add them to the comparison
            </Text>
          </YStack>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}
