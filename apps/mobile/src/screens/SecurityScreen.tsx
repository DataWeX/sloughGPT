import React, {useEffect, useState, useCallback} from 'react';
import {FlatList, Pressable, RefreshControl} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {YStack, XStack, Text} from 'tamagui';
import {useColors} from '../theme/colors';
import {api} from '../services/api-client';
import {Icon} from '../components/Icon';
import {StatusBadge} from '../components/StatusBadge';
import {triggerHaptic} from '../services/haptics';

interface AuditLog {
  event_type: string;
  timestamp: string;
  user?: string;
  resource?: string;
  detail?: string;
}

export function SecurityScreen() {
  const colors = useColors();
  const [logs, setLogs] = useState<AuditLog[]>([]);
  const [keyInfo, setKeyInfo] = useState<{count: number; configured: boolean} | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const fetchData = useCallback(async () => {
    try {
      const [auditRes, keyRes] = await Promise.all([
        api.get<{logs: AuditLog[]; count: number}>('/security/audit?limit=100').catch(() => ({logs: [], count: 0})),
        api.get<{count: number; configured: boolean}>('/security/keys').catch(() => ({count: 0, configured: false})),
      ]);
      setLogs(auditRes.logs ?? []);
      setKeyInfo(keyRes);
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

  const formatTime = (ts: string) => {
    const d = new Date(ts);
    const now = new Date();
    const diff = now.getTime() - d.getTime();
    if (diff < 60000) return 'just now';
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
    return d.toLocaleDateString();
  };

  const getEventVariant = (event: string) => {
    if (event.includes('delete') || event.includes('revoke')) return 'error' as const;
    if (event.includes('create') || event.includes('grant')) return 'success' as const;
    if (event.includes('update') || event.includes('rotate')) return 'info' as const;
    return 'default' as const;
  };

  const accent = colors.primary;

  return (
    <SafeAreaView style={{flex: 1, backgroundColor: colors.background}} edges={['top']}>
      <XStack paddingHorizontal={16} paddingVertical={12} alignItems="center" justifyContent="space-between">
        <YStack>
          <Text fontSize={20} fontWeight="600" color={colors.text}>Security</Text>
          <Text fontSize={12} color={colors.textSecondary}>
            {logs.length} audit events
          </Text>
        </YStack>
        <Pressable onPress={onRefresh} style={{padding: 8}}>
          <Icon name="refresh-cw" size={20} color={accent} />
        </Pressable>
      </XStack>

      {/* API Keys Card */}
      <YStack paddingHorizontal={16} marginBottom={12}>
        <YStack
          backgroundColor={colors.backgroundHover}
          borderRadius={8}
          padding={12}
          flexDirection="row"
          alignItems="center"
          justifyContent="space-between">
          <XStack alignItems="center" gap={8}>
            <Icon name="lock" size={18} color={accent} />
            <Text fontSize={13} fontWeight="500" color={colors.text}>API Keys</Text>
          </XStack>
          <StatusBadge
            label={keyInfo?.configured ? `${keyInfo.count} configured` : 'Not configured'}
            variant={keyInfo?.configured ? 'success' : 'warning'}
          />
        </YStack>
      </YStack>

      {/* Audit Logs */}
      {loading ? (
        <YStack flex={1} alignItems="center" justifyContent="center">
          <Icon name="refresh-cw" size={24} color={colors.textSecondary} />
          <Text fontSize={13} color={colors.textSecondary} marginTop={8}>Loading audit logs...</Text>
        </YStack>
      ) : (
        <FlatList
          data={logs}
          keyExtractor={(item, idx) => `${item.timestamp}-${idx}`}
          contentContainerStyle={{paddingHorizontal: 16, paddingBottom: 20}}
          refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={accent} />}
          ListEmptyComponent={
            <YStack alignItems="center" paddingVertical={40}>
              <Icon name="lock" size={32} color={colors.textSecondary} />
              <Text fontSize={14} color={colors.textSecondary} marginTop={8}>No audit events</Text>
            </YStack>
          }
          renderItem={({item}) => (
            <YStack
              backgroundColor={colors.backgroundHover}
              borderRadius={8}
              padding={12}
              marginBottom={8}
              borderLeftWidth={3}
              borderLeftColor={
                item.event_type.includes('delete') || item.event_type.includes('revoke')
                  ? colors.error
                  : item.event_type.includes('create')
                    ? colors.success
                    : accent
              }>
              <XStack justifyContent="space-between" alignItems="flex-start">
                <Text fontSize={13} fontWeight="500" color={colors.text} flex={1}>
                  {item.event_type}
                </Text>
                <StatusBadge label={item.event_type} variant={getEventVariant(item.event_type)} />
              </XStack>
              <XStack marginTop={6} gap={12}>
                {item.user && (
                  <XStack alignItems="center" gap={4}>
                    <Icon name="user" size={12} color={colors.textSecondary} />
                    <Text fontSize={11} color={colors.textSecondary}>{item.user}</Text>
                  </XStack>
                )}
                {item.resource && (
                  <XStack alignItems="center" gap={4}>
                    <Icon name="package" size={12} color={colors.textSecondary} />
                    <Text fontSize={11} color={colors.textSecondary}>{item.resource}</Text>
                  </XStack>
                )}
              </XStack>
              {item.detail && (
                <Text fontSize={11} color={colors.textMuted} marginTop={4} numberOfLines={2}>
                  {item.detail}
                </Text>
              )}
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
