import React, {useEffect, useState, useCallback} from 'react';
import {FlatList, Pressable, RefreshControl} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {YStack, XStack, Text} from 'tamagui';
import {useColors} from '../theme/colors';
import {api} from '../services/api-client';
import {Icon} from '../components/Icon';
import {StatusBadge} from '../components/StatusBadge';
import {triggerHaptic} from '../services/haptics';

interface RegisteredModel {
  model_id: string;
  status: string;
  metrics?: Record<string, unknown>;
  registered_at?: string;
}

interface RegistryStats {
  total_models: number;
  loaded_models: number;
  failed_models: number;
  circuit_breaker_open: boolean;
}

export function RegistryScreen() {
  const colors = useColors();
  const [models, setModels] = useState<RegisteredModel[]>([]);
  const [stats, setStats] = useState<RegistryStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      const [modelsRes, statsRes] = await Promise.all([
        api.get<{models?: RegisteredModel[]} | RegisteredModel[]>('/registry/models').catch(() => []),
        api.get<{models_loaded?: number; models_registered?: number; healthy?: boolean; degraded?: boolean; has_errors?: boolean}>('/registry/stats').catch(() => null),
      ]);
      const modelList = Array.isArray(modelsRes) ? modelsRes : (modelsRes?.models ?? []);
      setModels(modelList);
      if (statsRes) {
        setStats({
          total_models: statsRes.models_registered ?? 0,
          loaded_models: statsRes.models_loaded ?? 0,
          failed_models: statsRes.has_errors ? 1 : 0,
          circuit_breaker_open: statsRes.degraded ?? false,
        });
      }
    } catch {
      // handled above
    }
  }, []);

  useEffect(() => {
    fetchData().finally(() => setLoading(false));
  }, [fetchData]);

  const onRefresh = async () => {
    setRefreshing(true);
    await fetchData();
    setRefreshing(false);
  };

  const accent = colors.primary;

  const getStatusVariant = (status: string) => {
    if (status === 'loaded' || status === 'healthy') return 'success' as const;
    if (status === 'failed' || status === 'error') return 'error' as const;
    if (status === 'loading' || status === 'initializing') return 'info' as const;
    return 'default' as const;
  };

  return (
    <SafeAreaView style={{flex: 1, backgroundColor: colors.background}} edges={['top']}>
      <XStack paddingHorizontal={16} paddingVertical={12} alignItems="center" justifyContent="space-between">
        <YStack>
          <Text fontSize={20} fontWeight="600" color={colors.text}>Registry</Text>
          <Text fontSize={12} color={colors.textSecondary}>
            {stats ? `${stats.loaded_models}/${stats.total_models} loaded` : `${models.length} models`}
          </Text>
        </YStack>
        <Pressable onPress={onRefresh} style={{padding: 8}}>
          <Icon name="refresh-cw" size={20} color={accent} />
        </Pressable>
      </XStack>

      {/* Stats Cards */}
      {stats && (
        <XStack paddingHorizontal={16} gap={8} marginBottom={12}>
          <YStack flex={1} backgroundColor={colors.backgroundHover} borderRadius={8} padding={10} alignItems="center">
            <Text fontSize={18} fontWeight="700" color={accent}>{stats.total_models}</Text>
            <Text fontSize={10} color={colors.textSecondary}>Total</Text>
          </YStack>
          <YStack flex={1} backgroundColor={colors.backgroundHover} borderRadius={8} padding={10} alignItems="center">
            <Text fontSize={18} fontWeight="700" color={colors.success}>{stats.loaded_models}</Text>
            <Text fontSize={10} color={colors.textSecondary}>Loaded</Text>
          </YStack>
          <YStack flex={1} backgroundColor={colors.backgroundHover} borderRadius={8} padding={10} alignItems="center">
            <Text fontSize={18} fontWeight="700" color={colors.error}>{stats.failed_models}</Text>
            <Text fontSize={10} color={colors.textSecondary}>Failed</Text>
          </YStack>
          <YStack flex={1} backgroundColor={colors.backgroundHover} borderRadius={8} padding={10} alignItems="center">
            <StatusBadge
              label={stats.circuit_breaker_open ? 'Open' : 'Closed'}
              variant={stats.circuit_breaker_open ? 'error' : 'success'}
            />
            <Text fontSize={10} color={colors.textSecondary} marginTop={4}>Breaker</Text>
          </YStack>
        </XStack>
      )}

      {/* Model List */}
      {loading ? (
        <YStack flex={1} alignItems="center" justifyContent="center">
          <Icon name="refresh-cw" size={24} color={colors.textSecondary} />
          <Text fontSize={13} color={colors.textSecondary} marginTop={8}>Loading registry...</Text>
        </YStack>
      ) : (
        <FlatList
          data={models}
          keyExtractor={item => item.model_id}
          contentContainerStyle={{paddingHorizontal: 16, paddingBottom: 20}}
          refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={accent} />}
          ListEmptyComponent={
            <YStack alignItems="center" paddingVertical={40}>
              <Icon name="package" size={32} color={colors.textSecondary} />
              <Text fontSize={14} color={colors.textSecondary} marginTop={8}>No registered models</Text>
            </YStack>
          }
          renderItem={({item}) => {
            const isSelected = selectedId === item.model_id;
            return (
              <YStack marginBottom={8}>
                <Pressable
                  onPress={() => setSelectedId(isSelected ? null : item.model_id)}
                  style={{width: '100%'}}>
                  <YStack
                    backgroundColor={isSelected ? `${accent}15` : colors.backgroundHover}
                    borderRadius={8}
                    padding={12}
                    borderWidth={isSelected ? 1 : 0}
                    borderColor={accent}>
                    <XStack justifyContent="space-between" alignItems="center">
                      <YStack flex={1}>
                        <Text fontSize={13} fontWeight="500" color={colors.text} numberOfLines={1}>
                          {item.model_id}
                        </Text>
                        {item.registered_at && (
                          <Text fontSize={11} color={colors.textMuted} marginTop={2}>
                            Registered {new Date(item.registered_at).toLocaleDateString()}
                          </Text>
                        )}
                      </YStack>
                      <StatusBadge label={item.status} variant={getStatusVariant(item.status)} />
                    </XStack>
                  </YStack>
                </Pressable>

                {/* Expanded Details */}
                {isSelected && item.metrics && (
                  <YStack
                    backgroundColor={colors.backgroundHover}
                    borderRadius={8}
                    padding={12}
                    marginTop={4}
                    borderLeftWidth={3}
                    borderLeftColor={accent}>
                    <Text fontSize={12} fontWeight="500" color={colors.textSecondary} marginBottom={4}>Metrics</Text>
                    {Object.entries(item.metrics).map(([key, value]) => (
                      <XStack key={key} justifyContent="space-between" paddingVertical={2}>
                        <Text fontSize={12} color={colors.text}>{key}</Text>
                        <Text fontSize={12} color={colors.textMuted}>
                          {typeof value === 'number' ? value.toFixed(4) : String(value)}
                        </Text>
                      </XStack>
                    ))}
                  </YStack>
                )}
              </YStack>
            );
          }}
        />
      )}
    </SafeAreaView>
  );
}
