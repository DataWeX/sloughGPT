import React, {useEffect, useState, useCallback} from 'react';
import {ScrollView, Pressable, RefreshControl} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {YStack, XStack, Text} from 'tamagui';
import {useRoute, useNavigation} from '@react-navigation/native';
import type {StackNavigationProp} from '@react-navigation/stack';
import type {ToolsStackParamList} from '../navigation/types';
import {useColors} from '../theme/colors';
import {api} from '../services/api-client';
import {Icon} from '../components/Icon';
import {StatusBadge} from '../components/StatusBadge';
import {triggerHaptic} from '../services/haptics';
import {toast} from '../services/toast';
import {useModelStore} from '../stores/model-store';
import type {ModelInfo, BenchmarkResult} from '../types';

export function ModelDetailScreen() {
  const route = useRoute();
  const navigation = useNavigation<StackNavigationProp<ToolsStackParamList>>();
  const colors = useColors();
  const {modelId} = route.params as {modelId: string};
  const {models, currentModel, health, loadModel, unloadModel, loadingModelId} = useModelStore();
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [benchmark, setBenchmark] = useState<BenchmarkResult | null>(null);
  const [benchRunning, setBenchRunning] = useState(false);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const fetchModel = useCallback(async () => {
    try {
      const allModels = await api.get<ModelInfo[]>('/models').catch(() => []);
      const found = allModels.find(m => m.id === modelId || m.name === modelId);
      setModel(found || null);
    } catch {
      setModel(null);
    }
  }, [modelId]);

  useEffect(() => {
    fetchModel().finally(() => setLoading(false));
  }, [fetchModel]);

  const onRefresh = async () => {
    setRefreshing(true);
    await fetchModel();
    setRefreshing(false);
  };

  const isLoaded = currentModel === modelId || (model?.loaded ?? false);
  const isLoading = loadingModelId === modelId;

  const handleLoad = async () => {
    try {
      triggerHaptic('light');
      await loadModel(modelId);
      triggerHaptic('success');
      toast.success(`Loaded ${model?.name || modelId}`);
      await fetchModel();
    } catch {
      toast.error('Failed to load model');
    }
  };

  const handleUnload = async () => {
    try {
      triggerHaptic('light');
      await unloadModel();
      triggerHaptic('success');
      toast.success('Model unloaded');
      await fetchModel();
    } catch {
      toast.error('Failed to unload model');
    }
  };

  const handleBenchmark = async () => {
    try {
      setBenchRunning(true);
      triggerHaptic('light');
      const result = await api.post<BenchmarkResult>('/benchmark/run', {model: modelId});
      setBenchmark(result);
      triggerHaptic('success');
      toast.success('Benchmark complete');
    } catch {
      toast.error('Benchmark failed');
    } finally {
      setBenchRunning(false);
    }
  };

  return (
    <SafeAreaView style={{flex: 1, backgroundColor: colors.background}} edges={['top']}>
      <XStack paddingHorizontal={16} paddingVertical={12} alignItems="center" gap={12}>
        <Pressable onPress={() => navigation.goBack()}>
          <Icon name="arrow-down" size={20} color={colors.textMuted} />
        </Pressable>
        <Text fontSize={18} fontWeight="600" color={colors.text} flex={1} numberOfLines={1}>
          {model?.name || modelId}
        </Text>
        <Pressable onPress={onRefresh}>
          <Icon name="refresh-cw" size={18} color={colors.primary} />
        </Pressable>
      </XStack>

      {loading ? (
        <YStack flex={1} alignItems="center" justifyContent="center">
          <StatusBadge label="Loading..." variant="info" />
        </YStack>
      ) : (
        <ScrollView
          contentContainerStyle={{paddingBottom: 32}}
          refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}>
          <YStack padding={16} gap={12}>
              {/* Status */}
              <YStack padding={14} borderRadius={10} backgroundColor={colors.card} borderWidth={0.5} borderColor={colors.border} gap={10}>
                <XStack justifyContent="space-between" alignItems="center">
                  <Text fontSize={15} fontWeight="600" color={colors.text}>Model Info</Text>
                  <StatusBadge label={isLoaded ? 'Loaded' : 'Not Loaded'} variant={isLoaded ? 'success' : 'default'} />
                </XStack>
                {model && (
                  <YStack gap={6}>
                    {[
                      {label: 'Name', value: model.name},
                      {label: 'Type', value: model.type},
                      {label: 'Parameters', value: model.params},
                      {label: 'Source', value: model.source},
                      {label: 'Size', value: model.size_gb ? `${model.size_gb.toFixed(2)} GB` : model.size_mb ? `${model.size_mb} MB` : '—'},
                    ].map(item => (
                      <XStack key={item.label} justifyContent="space-between" alignItems="center">
                        <Text fontSize={13} color={colors.textMuted}>{item.label}</Text>
                        <Text fontSize={13} fontWeight="500" color={colors.text} numberOfLines={1} maxWidth={200}>{item.value || '—'}</Text>
                      </XStack>
                    ))}
                  </YStack>
                )}
                {model?.description && (
                  <YStack gap={4}>
                    <Text fontSize={12} color={colors.textMuted}>Description</Text>
                    <Text fontSize={13} color={colors.text} lineHeight={18}>{model.description}</Text>
                  </YStack>
                )}
              </YStack>

              {/* Actions */}
              <YStack padding={14} borderRadius={10} backgroundColor={colors.card} borderWidth={0.5} borderColor={colors.border} gap={8}>
                <Text fontSize={15} fontWeight="600" color={colors.text}>Actions</Text>
                <XStack gap={8}>
                  {!isLoaded ? (
                    <Pressable onPress={handleLoad} disabled={isLoading} style={{flex: 1}}>
                      <XStack padding={10} borderRadius={8} backgroundColor={!isLoading ? colors.primary : colors.border} alignItems="center" justifyContent="center" gap={6}>
                        <Icon name={isLoading ? 'refresh-cw' : 'download'} size={16} color="white" />
                        <Text fontSize={13} fontWeight="600" color="white">{isLoading ? 'Loading...' : 'Load'}</Text>
                      </XStack>
                    </Pressable>
                  ) : (
                    <Pressable onPress={handleUnload} style={{flex: 1}}>
                      <XStack padding={10} borderRadius={8} backgroundColor={colors.error} alignItems="center" justifyContent="center" gap={6}>
                        <Icon name="trash-2" size={16} color="white" />
                        <Text fontSize={13} fontWeight="600" color="white">Unload</Text>
                      </XStack>
                    </Pressable>
                  )}
                  <Pressable onPress={handleBenchmark} disabled={!isLoaded || benchRunning} style={{flex: 1}}>
                    <XStack padding={10} borderRadius={8} backgroundColor={isLoaded && !benchRunning ? colors.primaryAlpha(0.1) : colors.background} alignItems="center" justifyContent="center" gap={6}>
                      <Icon name="bar-chart" size={16} color={isLoaded ? colors.primary : colors.textMuted} />
                      <Text fontSize={13} fontWeight="500" color={isLoaded ? colors.primary : colors.textMuted}>{benchRunning ? 'Running...' : 'Benchmark'}</Text>
                    </XStack>
                  </Pressable>
                </XStack>
              </YStack>

              {/* Tags */}
              {model?.tags && model.tags.length > 0 && (
                <YStack padding={14} borderRadius={10} backgroundColor={colors.card} borderWidth={0.5} borderColor={colors.border} gap={6}>
                  <Text fontSize={15} fontWeight="600" color={colors.text}>Tags</Text>
                  <XStack gap={4} flexWrap="wrap">
                    {model.tags.map(tag => (
                      <StatusBadge key={tag} label={tag} variant="info" />
                    ))}
                  </XStack>
                </YStack>
              )}

              {/* Benchmark Results */}
              {benchmark && (
                <YStack padding={14} borderRadius={10} backgroundColor={colors.card} borderWidth={0.5} borderColor={colors.border} gap={8}>
                  <Text fontSize={15} fontWeight="600" color={colors.text}>Benchmark Results</Text>
                  <XStack gap={12}>
                    {[
                      {label: 'Coherence', value: benchmark.coherence.toFixed(2)},
                      {label: 'Repetition', value: `${(benchmark.repetition * 100).toFixed(1)}%`},
                      {label: 'Perplexity', value: benchmark.perplexity != null ? String(benchmark.perplexity) : '—'},
                      {label: 'Avg Length', value: String(benchmark.avg_length ?? 0)},
                    ].map(item => (
                      <YStack key={item.label} flex={1} alignItems="center" gap={2}>
                        <Text fontSize={16} fontWeight="700" color={colors.primary}>{item.value}</Text>
                        <Text fontSize={10} color={colors.textMuted}>{item.label}</Text>
                      </YStack>
                    ))}
                  </XStack>
                </YStack>
              )}

              {/* Server Health */}
              {health && (
                <YStack padding={12} borderRadius={10} backgroundColor={colors.card} borderWidth={0.5} borderColor={colors.border} gap={4}>
                  <XStack justifyContent="space-between" alignItems="center">
                    <Text fontSize={13} color={colors.textMuted}>Server</Text>
                    <StatusBadge label={health.status} variant={health.status === 'healthy' ? 'success' : 'error'} />
                  </XStack>
                  <XStack justifyContent="space-between" alignItems="center">
                    <Text fontSize={13} color={colors.textMuted}>Inference Count</Text>
                    <Text fontSize={13} fontWeight="500" color={colors.text}>{health.inference_count}</Text>
                  </XStack>
                </YStack>
              )}
            </YStack>
          </ScrollView>
      )}
    </SafeAreaView>
  );
}
