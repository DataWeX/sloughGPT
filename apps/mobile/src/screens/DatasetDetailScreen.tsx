import React, {useEffect, useState, useCallback} from 'react';
import {FlatList, Pressable, RefreshControl} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {YStack, XStack, Text} from 'tamagui';
import {useRoute, useNavigation} from '@react-navigation/native';
import type {NativeStackNavigationProp} from '@react-navigation/native-stack';
import type {ToolsStackParamList} from '../navigation/types';
import {useColors} from '../theme/colors';
import {api} from '../services/api-client';
import {Icon} from '../components/Icon';
import {StatusBadge} from '../components/StatusBadge';
import {triggerHaptic} from '../services/haptics';
import {toast} from '../services/toast';

interface Dataset {
  id: string;
  name: string;
  description: string;
  row_count: number;
  total_chars: number;
  format: string;
  source: string;
  tags: string[];
  created_at: string;
  updated_at: string;
}

interface DatasetStats {
  total_rows: number;
  avg_length: number;
  total_chars: number;
  format: string;
}

interface DatasetPreview {
  headers: string[];
  rows: string[][];
}

export function DatasetDetailScreen() {
  const route = useRoute();
  const navigation = useNavigation<NativeStackNavigationProp<ToolsStackParamList>>();
  const colors = useColors();
  const {datasetId} = route.params as {datasetId: string};
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [stats, setStats] = useState<DatasetStats | null>(null);
  const [preview, setPreview] = useState<DatasetPreview | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [deleting, setDeleting] = useState(false);

  const fetchDataset = useCallback(async () => {
    try {
      const [ds, st, pv] = await Promise.all([
        api.get<Dataset>(`/datasets/${datasetId}`).catch(() => null),
        api.get<DatasetStats>(`/datasets/${datasetId}/stats`).catch(() => null),
        api.get<DatasetPreview>(`/datasets/${datasetId}/preview`).catch(() => null),
      ]);
      setDataset(ds);
      setStats(st);
      setPreview(pv);
    } catch {
      // handled above
    }
  }, [datasetId]);

  useEffect(() => {
    fetchDataset().finally(() => setLoading(false));
  }, [fetchDataset]);

  const onRefresh = async () => {
    setRefreshing(true);
    await fetchDataset();
    setRefreshing(false);
  };

  const handleDelete = async () => {
    try {
      setDeleting(true);
      triggerHaptic('light');
      await api.delete(`/datasets/${datasetId}`);
      triggerHaptic('success');
      toast.success('Dataset deleted');
    } catch {
      toast.error('Failed to delete dataset');
    } finally {
      setDeleting(false);
    }
  };

  const formatBytes = (chars: number): string => {
    if (chars < 1024) return `${chars} B`;
    if (chars < 1024 * 1024) return `${(chars / 1024).toFixed(1)} KB`;
    return `${(chars / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <SafeAreaView style={{flex: 1, backgroundColor: colors.background}} edges={['top']}>
      <XStack paddingHorizontal={16} paddingVertical={12} alignItems="center" gap={12}>
        <Pressable onPress={() => navigation.goBack()}>
          <Icon name="chevron-down" size={20} color={colors.textMuted} />
        </Pressable>
        <Text fontSize={18} fontWeight="600" color={colors.text} flex={1} numberOfLines={1}>
          {dataset?.name || datasetId}
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
        <FlatList
          data={[]}
          renderItem={() => null}
          ListHeaderComponent={
            <YStack padding={16} gap={12}>
              {/* Info */}
              <YStack padding={14} borderRadius={10} backgroundColor={colors.white} borderWidth={0.5} borderColor={colors.border} gap={8}>
                <XStack justifyContent="space-between" alignItems="center">
                  <Text fontSize={15} fontWeight="600" color={colors.text}>Dataset Info</Text>
                  <StatusBadge label={dataset?.format || 'unknown'} variant="info" />
                </XStack>
                {dataset?.description && (
                  <Text fontSize={13} color={colors.textMuted} lineHeight={18}>{dataset.description}</Text>
                )}
                {dataset?.source && (
                  <XStack justifyContent="space-between" alignItems="center">
                    <Text fontSize={13} color={colors.textMuted}>Source</Text>
                    <Text fontSize={13} fontWeight="500" color={colors.text}>{dataset.source}</Text>
                  </XStack>
                )}
                {dataset?.tags && dataset.tags.length > 0 && (
                  <XStack gap={4} flexWrap="wrap">
                    {dataset.tags.map(tag => (
                      <StatusBadge key={tag} label={tag} variant="info" />
                    ))}
                  </XStack>
                )}
              </YStack>

              {/* Stats */}
              <YStack padding={14} borderRadius={10} backgroundColor={colors.white} borderWidth={0.5} borderColor={colors.border} gap={8}>
                <Text fontSize={15} fontWeight="600" color={colors.text}>Statistics</Text>
                <XStack gap={8}>
                  {[
                    {label: 'Rows', value: String(stats?.total_rows ?? dataset?.row_count ?? 0)},
                    {label: 'Avg Length', value: String(stats?.avg_length ?? 0)},
                    {label: 'Total Size', value: formatBytes(stats?.total_chars ?? dataset?.total_chars ?? 0)},
                  ].map(item => (
                    <YStack key={item.label} flex={1} padding={10} borderRadius={8} backgroundColor={colors.background} alignItems="center" gap={2}>
                      <Text fontSize={16} fontWeight="700" color={colors.primary}>{item.value}</Text>
                      <Text fontSize={10} color={colors.textMuted}>{item.label}</Text>
                    </YStack>
                  ))}
                </XStack>
              </YStack>

              {/* Preview */}
              {preview && preview.rows && preview.rows.length > 0 && (
                <YStack padding={14} borderRadius={10} backgroundColor={colors.white} borderWidth={0.5} borderColor={colors.border} gap={8}>
                  <Text fontSize={15} fontWeight="600" color={colors.text}>Preview</Text>
                  <YStack padding={8} borderRadius={6} backgroundColor={colors.background}>
                    {preview.rows.slice(0, 5).map((row, i) => (
                      <Text key={i} fontSize={11} fontFamily="monospace" color={colors.text} numberOfLines={2}>
                        {row.join(' | ')}
                      </Text>
                    ))}
                    {preview.rows.length > 5 && (
                      <Text fontSize={11} color={colors.textMuted}>... and {preview.rows.length - 5} more rows</Text>
                    )}
                  </YStack>
                </YStack>
              )}

              {/* Actions */}
              <YStack padding={14} borderRadius={10} backgroundColor={colors.white} borderWidth={0.5} borderColor={colors.border} gap={8}>
                <Text fontSize={15} fontWeight="600" color={colors.text}>Actions</Text>
                <Pressable onPress={handleDelete} disabled={deleting}>
                  <XStack padding={10} borderRadius={8} backgroundColor={colors.error} alignItems="center" justifyContent="center" gap={6}>
                    <Icon name="trash-2" size={16} color="white" />
                    <Text fontSize={13} fontWeight="600" color="white">{deleting ? 'Deleting...' : 'Delete Dataset'}</Text>
                  </XStack>
                </Pressable>
              </YStack>
            </YStack>
          }
          contentContainerStyle={{paddingBottom: 32}}
          refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}
        />
      )}
    </SafeAreaView>
  );
}
