import React, {useEffect, useState, useCallback, useRef} from 'react';
import {FlatList, Pressable, RefreshControl, AppState, Modal} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {YStack, XStack, Text, ScrollView} from 'tamagui';
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
  stack?: string;
}

interface ErrorDetail {
  fingerprint: string;
  message: string;
  source: string;
  count: number;
  sample_url: string;
  latest_timestamp: string;
  entries: Array<{
    message: string;
    url: string;
    line: number;
    col: number;
    client_host: string;
    timestamp: string;
    stack: string;
    metadata: Record<string, unknown>;
  }>;
}

type Tab = 'grouped' | 'recent';

const POLL_INTERVAL = 15000;

export function ErrorsScreen() {
  const colors = useColors();
  const [tab, setTab] = useState<Tab>('grouped');
  const [grouped, setGrouped] = useState<GroupedError[]>([]);
  const [recent, setRecent] = useState<RecentError[]>([]);
  const [total, setTotal] = useState(0);
  const [unread, setUnread] = useState(0);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [clearing, setClearing] = useState(false);
  const [detailError, setDetailError] = useState<ErrorDetail | null>(null);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const appState = useRef(AppState.currentState);

  const fetchData = useCallback(async () => {
    try {
      const [g, r, u] = await Promise.all([
        api.get<{groups: GroupedError[]}>('/errors/grouped').catch(() => ({groups: []})),
        api.get<{errors: RecentError[]; total: number}>('/errors/recent?limit=50').catch(() => ({errors: [], total: 0})),
        api.get<{count: number}>('/errors/unread').catch(() => ({count: 0})),
      ]);
      setGrouped(g.groups ?? []);
      setRecent(r.errors ?? []);
      setTotal(r.total ?? 0);
      setUnread(u.count ?? 0);
    } catch {
      // handled above
    }
  }, []);

  useEffect(() => {
    fetchData().finally(() => setLoading(false));
  }, [fetchData]);

  useEffect(() => {
    const subscription = AppState.addEventListener('change', next => {
      if (appState.current.match(/inactive|background/) && next === 'active') {
        fetchData();
      }
      appState.current = next;
    });

    pollRef.current = setInterval(() => {
      if (AppState.currentState === 'active') {
        fetchData();
      }
    }, POLL_INTERVAL);

    return () => {
      subscription.remove();
      if (pollRef.current) clearInterval(pollRef.current);
    };
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

  const handleShowDetail = async (error: GroupedError) => {
    try {
      setLoadingDetail(true);
      const detail = await api.get<ErrorDetail>(`/errors/grouped/${error.fingerprint}`);
      setDetailError(detail);
    } catch {
      toast.error('Failed to load error detail');
    } finally {
      setLoadingDetail(false);
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
          <XStack alignItems="center" gap={8}>
            <Text fontSize={20} fontWeight="600" color={colors.text}>Errors</Text>
            {unread > 0 && (
              <XStack paddingHorizontal={6} paddingVertical={2} borderRadius={10} backgroundColor={colors.error}>
                <Text fontSize={10} fontWeight="700" color="white">{unread > 99 ? '99+' : unread}</Text>
              </XStack>
            )}
          </XStack>
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
              color={tab === t ? colors.white : colors.text}>
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
            <Pressable onPress={() => handleShowDetail(item)}>
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
                {item.sample_url ? (
                  <Text fontSize={10} color={colors.textMuted} marginTop={4} numberOfLines={1}>
                    {item.sample_url}
                  </Text>
                ) : null}
              </YStack>
            </Pressable>
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

      {/* Error Detail Modal */}
      <Modal visible={detailError !== null} animationType="slide" transparent>
        <YStack flex={1} backgroundColor="rgba(0,0,0,0.5)" justifyContent="flex-end">
          <YStack
            backgroundColor={colors.background}
            borderTopLeftRadius={20}
            borderTopRightRadius={20}
            maxHeight="80%"
            padding={16}>
            <XStack justifyContent="space-between" alignItems="center" marginBottom={12}>
              <Text fontSize={16} fontWeight="600" color={colors.text}>Error Detail</Text>
              <Pressable onPress={() => setDetailError(null)} style={{padding: 4}}>
                <Icon name="x" size={20} color={colors.textMuted} />
              </Pressable>
            </XStack>

            {loadingDetail ? (
              <YStack alignItems="center" paddingVertical={20}>
                <Icon name="refresh-cw" size={20} color={colors.textSecondary} />
              </YStack>
            ) : detailError && (
              <ScrollView style={{maxHeight: 500}}>
                <YStack gap={12}>
                  <YStack gap={4}>
                    <Text fontSize={13} fontWeight="500" color={colors.text}>{detailError.message}</Text>
                    <XStack gap={8}>
                      <StatusBadge label={detailError.source} variant="info" />
                      <StatusBadge label={`${detailError.count} occurrences`} variant="default" />
                    </XStack>
                  </YStack>

                  {detailError.entries?.slice(0, 3).map((entry, i) => (
                    <YStack key={i} padding={10} borderRadius={6} backgroundColor={colors.backgroundHover} gap={4}>
                      <XStack gap={8}>
                        {entry.url ? (
                          <Text fontSize={11} color={colors.textSecondary} numberOfLines={1} flex={1}>
                            {entry.url}:{entry.line}:{entry.col}
                          </Text>
                        ) : null}
                        <Text fontSize={11} color={colors.textMuted}>{formatTime(entry.timestamp)}</Text>
                      </XStack>
                      {entry.stack ? (
                        <Text fontSize={10} color={colors.textMuted} fontFamily="monospace" numberOfLines={4}>
                          {entry.stack}
                        </Text>
                      ) : null}
                      {entry.client_host ? (
                        <Text fontSize={10} color={colors.textMuted}>From: {entry.client_host}</Text>
                      ) : null}
                    </YStack>
                  ))}
                </YStack>
              </ScrollView>
            )}
          </YStack>
        </YStack>
      </Modal>
    </SafeAreaView>
  );
}
