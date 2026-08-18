import React, {useEffect, useState, useCallback} from 'react';
import {FlatList, Pressable, RefreshControl, Alert} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {YStack, XStack, Text} from 'tamagui';
import {useColors} from '../theme/colors';
import {api} from '../services/api-client';
import {Icon} from '../components/Icon';
import {StatusBadge} from '../components/StatusBadge';
import {triggerHaptic} from '../services/haptics';
import {toast} from '../services/toast';
import type {Adapter} from '../types';

export function AdaptersScreen() {
  const colors = useColors();
  const [adapters, setAdapters] = useState<Adapter[]>([]);
  const [aggregating, setAggregating] = useState(false);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const fetchAdapters = useCallback(async () => {
    try {
      const data = await api.get<{adapters: Adapter[]}>('/user-adapters');
      setAdapters(data.adapters || []);
    } catch {
      setAdapters([]);
    }
  }, []);

  useEffect(() => {
    fetchAdapters().finally(() => setLoading(false));
  }, [fetchAdapters]);

  const onRefresh = async () => {
    setRefreshing(true);
    await fetchAdapters();
    setRefreshing(false);
  };

  const handleAggregate = async () => {
    try {
      setAggregating(true);
      triggerHaptic('light');
      const result = await api.post<{verdict: string; delta?: Record<string, number>}>('/user-adapters/aggregate-best', {});
      triggerHaptic('success');
      toast.success(`Aggregated: ${result.verdict}`);
      await fetchAdapters();
    } catch {
      toast.error('Aggregation failed');
    } finally {
      setAggregating(false);
    }
  };

  const handleReset = (userId: string) => {
    Alert.alert('Reset Adapter', `Reset adapter for user ${userId}?`, [
      {text: 'Cancel', style: 'cancel'},
      {
        text: 'Reset',
        style: 'destructive',
        onPress: async () => {
          try {
            await api.post(`/user-adapters/${userId}/reset`);
            triggerHaptic('success');
            toast.success('Adapter reset');
            await fetchAdapters();
          } catch {
            toast.error('Reset failed');
          }
        },
      },
    ]);
  };

  const formatLoss = (loss: number) => loss?.toFixed(4) ?? '—';

  const renderItem = ({item}: {item: Adapter}) => (
    <XStack
      padding={12}
      borderRadius={10}
      borderWidth={0.5}
      borderColor={colors.border}
      backgroundColor={colors.white}
      gap={10}
      alignItems="center">
      <YStack width={36} height={36} borderRadius={8} backgroundColor={colors.primary + '15'} alignItems="center" justifyContent="center">
        <Icon name="layers" size={18} color={colors.primary} />
      </YStack>
      <YStack flex={1} gap={2}>
        <Text fontSize={14} fontWeight="500" color={colors.text}>{item.name || item.user_id || 'Unknown'}</Text>
        <XStack gap={6}>
          {item.loss != null && <StatusBadge label={`loss: ${formatLoss(item.loss)}`} variant="default" />}
          {item.steps != null && item.steps > 0 && <StatusBadge label={`${item.steps} steps`} variant="info" />}
        </XStack>
        {item.traits && Object.keys(item.traits).length > 0 && (
          <XStack gap={4} flexWrap="wrap">
            {Object.entries(item.traits).slice(0, 3).map(([k, v]) => (
              <StatusBadge key={k} label={`${k}: ${(v as number).toFixed(2)}`} variant="default" />
            ))}
          </XStack>
        )}
      </YStack>
      <Pressable onPress={() => item.user_id && handleReset(item.user_id)}>
        <Icon name="refresh-cw" size={16} color={colors.error} />
      </Pressable>
    </XStack>
  );

  return (
    <SafeAreaView style={{flex: 1, backgroundColor: colors.background}} edges={['top']}>
      <XStack paddingHorizontal={16} paddingVertical={12} alignItems="center" justifyContent="space-between">
        <Text fontSize={20} fontWeight="600" color={colors.text}>Adapters</Text>
        <Pressable onPress={handleAggregate} disabled={aggregating}>
          <XStack paddingHorizontal={10} paddingVertical={5} borderRadius={6} backgroundColor={aggregating ? colors.border : colors.primary} gap={4} alignItems="center">
            <Icon name={aggregating ? 'refresh-cw' : 'layers'} size={14} color="white" />
            <Text fontSize={12} fontWeight="500" color="white">{aggregating ? 'Aggregating...' : 'Aggregate'}</Text>
          </XStack>
        </Pressable>
      </XStack>

      {loading ? (
        <YStack flex={1} alignItems="center" justifyContent="center">
          <StatusBadge label="Loading..." variant="info" />
        </YStack>
      ) : adapters.length === 0 ? (
        <YStack flex={1} alignItems="center" justifyContent="center" gap={8}>
          <Icon name="layers" size={32} color={colors.textMuted} />
          <Text fontSize={14} color={colors.textMuted}>No adapters</Text>
          <Text fontSize={12} color={colors.textMuted}>Adapters are created from feedback</Text>
        </YStack>
      ) : (
        <FlatList
          data={adapters}
          keyExtractor={item => item.id}
          renderItem={renderItem}
          contentContainerStyle={{padding: 16, gap: 8}}
          refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}
        />
      )}
    </SafeAreaView>
  );
}
