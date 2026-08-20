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

interface GroupedError {
  fingerprint: string;
  message: string;
  source: string;
  count: number;
  latest: string;
  sample_url: string;
}

interface RecentError {
  id: string;
  message: string;
  source: string;
  url?: string;
  line?: number;
  timestamp: string;
}

type Tab = 'grouped' | 'recent';

export function ErrorsScreen() {
  const colors = useColors();
  const [tab, setTab] = useState<Tab>('grouped');
  const [grouped, setGrouped] = useState<GroupedError[]>([]);
  const [recent, setRecent] = useState<RecentError[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [clearing, setClearing] = useState(false);

  const fetchData = useCallback(async () => {
    try {
      const [g, r] = await Promise.all([
        api.get<{groups: GroupedError[]}>('/errors/grouped').catch(() => ({groups: []})),
        api.get<{errors: RecentError[]; total: number}>('/errors/recent?limit=50').catch(() => ({errors: [], total: 0})),
      ]);
      setGrouped(g.groups ?? []);
      setRecent(r.errors ?? []);
      setTotal(r.total ?? 0);
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

  const handleClear = async () => {
    try {
      setClearing(true);
      triggerHaptic('light');
      await api.delete('/errors/clear');
      triggerHaptic('success');
      toast.success('Errors cleared');
      await fetchData();
    } catch {
      toast.error('Failed to clear errors');
    } finally {
      setClearing(false);
    }
  };

  const formatTime = (ts: string) => {
    const d = new Date(ts);
    const now = new Date();
    const diff = now.getTime() - d.getTime();
    if (diff < 60000) return 'just now';
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
    return d.toLocaleDateString();
  };

  const accent = colors.primary;

  return (
    <SafeAreaView style={{flex: 1, backgroundColor: colors.background}} edges={['top']}>
      <XStack paddingHorizontal={16} paddingVertical={12} alignItems="center" justifyContent="space-between">
        <YStack>
          <Text fontSize={20} fontWeight="600" color={colors.text}>Errors</Text>
          <Text fontSize={12} color={colors.textSecondary}>
            {total > 0 ? `${total} total errors` : 'No errors'}
          </Text>
        </YStack>
        <XStack gap={8}>
          <Pressable
            onPress={onRefresh}
            style={{padding: 8}}>
            <Icon name="refresh-cw" size={20} color={accent} />
          </Pressable>
          <Pressable
            onPress={handleClear}
            disabled={clearing || total === 0}
            style={{padding: 8, opacity: clearing || total === 0 ? 0.5 : 1}}>
            <Icon name="trash-2" size={20} color={colors.error} />
          </Pressable>
        </XStack>
      </XStack>

      {/* Tabs */}
      <XStack paddingHorizontal={16} gap={4} marginBottom={12}>
        {(['grouped', 'recent'] as Tab[]).map(t => (
          <Pressable
            key={t}
            onPress={() => {
              setTab(t);
              triggerHaptic('light');
            }}
            style={{
              paddingHorizontal: 16,
              paddingVertical: 8,
              borderRadius: 8,
              backgroundColor: tab === t ? accent : colors.background,
            }}>
            <Text
              fontSize={13}
              fontWeight={tab === t ? '600' : '400'}
              color={tab === t ? '#fff' : colors.text}>
              {t === 'grouped' ? 'Grouped' : 'Recent'}
            </Text>
          </Pressable>
        ))}
      </XStack>

      {loading ? (
        <YStack flex={1} alignItems="center" justifyContent="center">
          <Icon name="refresh-cw" size={24} color={colors.textSecondary} />
          <Text fontSize={13} color={colors.textSecondary} marginTop={8}>Loading errors...</Text>
        </YStack>
      ) : tab === 'grouped' ? (
        <FlatList
          data={grouped}
          keyExtractor={item => item.fingerprint}
          contentContainerStyle={{paddingHorizontal: 16, paddingBottom: 20}}
          refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={accent} />}
          ListEmptyComponent={
            <YStack alignItems="center" paddingVertical={40}>
              <Icon name="check" size={32} color={colors.success} />
              <Text fontSize={14} color={colors.textSecondary} marginTop={8}>No errors logged</Text>
            </YStack>
          }
          renderItem={({item}) => (
            <YStack
              backgroundColor={colors.backgroundHover}
              borderRadius={8}
              padding={12}
              marginBottom={8}
              borderLeftWidth={3}
              borderLeftColor={item.count > 10 ? colors.error : item.count > 3 ? colors.warning : colors.success}>
              <XStack justifyContent="space-between" alignItems="flex-start">
                <Text fontSize={13} fontWeight="500" color={colors.text} flex={1} numberOfLines={2}>
                  {item.message}
                </Text>
                <StatusBadge
                  label={`${item.count}x`}
                  variant={item.count > 10 ? 'error' : item.count > 3 ? 'warning' : 'default'}
                />
              </XStack>
              <XStack marginTop={6} gap={12}>
                <Text fontSize={11} color={colors.textSecondary}>{item.source}</Text>
                <Text fontSize={11} color={colors.textSecondary}>{formatTime(item.latest)}</Text>
              </XStack>
            </YStack>
          )}
        />
      ) : (
        <FlatList
          data={recent}
          keyExtractor={item => item.id}
          contentContainerStyle={{paddingHorizontal: 16, paddingBottom: 20}}
          refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={accent} />}
          ListEmptyComponent={
            <YStack alignItems="center" paddingVertical={40}>
              <Icon name="check" size={32} color={colors.success} />
              <Text fontSize={14} color={colors.textSecondary} marginTop={8}>No recent errors</Text>
            </YStack>
          }
          renderItem={({item}) => (
            <YStack
              backgroundColor={colors.backgroundHover}
              borderRadius={8}
              padding={12}
              marginBottom={8}>
              <Text fontSize={13} fontWeight="500" color={colors.text} numberOfLines={2}>
                {item.message}
              </Text>
              <XStack marginTop={6} gap={12}>
                <Text fontSize={11} color={colors.textSecondary}>{item.source}</Text>
                {item.url && (
                  <Text fontSize={11} color={colors.textSecondary} numberOfLines={1} flex={1}>
                    {item.url}:{item.line ?? ''}
                  </Text>
                )}
              </XStack>
              <Text fontSize={11} color={colors.textMuted} marginTop={4}>
                {formatTime(item.timestamp)}
              </Text>
            </YStack>
          )}
        />
      )}
    </SafeAreaView>
  );
}
