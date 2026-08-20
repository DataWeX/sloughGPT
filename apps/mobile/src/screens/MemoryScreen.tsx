import React, {useEffect, useState, useCallback} from 'react';
import {FlatList, Pressable, TextInput as RNTextInput, RefreshControl, Alert} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {YStack, XStack, Text} from 'tamagui';
import {useColors} from '../theme/colors';
import {api} from '../services/api-client';
import {Icon} from '../components/Icon';
import {StatusBadge} from '../components/StatusBadge';
import {triggerHaptic} from '../services/haptics';
import {toast} from '../services/toast';

interface MemoryItem {
  id: string;
  content: string;
  topic: string;
  importance: number;
  source: string;
  created_at: string;
}

interface MemoryStats {
  total_items: number;
  topics: string[];
  enabled: boolean;
}

type Tab = 'all' | 'store';

export function MemoryScreen() {
  const colors = useColors();
  const [tab, setTab] = useState<Tab>('all');
  const [items, setItems] = useState<MemoryItem[]>([]);
  const [stats, setStats] = useState<MemoryStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [searching, setSearching] = useState(false);

  // Store form
  const [storeContent, setStoreContent] = useState('');
  const [storeTopic, setStoreTopic] = useState('');
  const [storing, setStoring] = useState(false);

  const fetchData = useCallback(async () => {
    try {
      const [listRes, statsRes] = await Promise.all([
        api.get<{items: MemoryItem[]}>('/memory/list').catch(() => ({items: []})),
        api.get<MemoryStats>('/memory/stats').catch(() => null),
      ]);
      setItems(listRes.items ?? []);
      setStats(statsRes);
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

  const handleSearch = async () => {
    if (!searchQuery.trim()) {
      await fetchData();
      return;
    }
    try {
      setSearching(true);
      const data = await api.get<{items: MemoryItem[]}>(`/memory/search?q=${encodeURIComponent(searchQuery.trim())}`).catch(() => ({items: []}));
      setItems(data.items ?? []);
    } catch {
      // handled above
    } finally {
      setSearching(false);
    }
  };

  const handleStore = async () => {
    if (!storeContent.trim()) return;
    try {
      setStoring(true);
      triggerHaptic('light');
      await api.post('/memory/store', {
        content: storeContent.trim(),
        topic: storeTopic.trim() || 'manual',
        source: 'mobile',
      });
      triggerHaptic('success');
      toast.success('Memory stored');
      setStoreContent('');
      setStoreTopic('');
      setTab('all');
      await fetchData();
    } catch {
      toast.error('Failed to store memory');
    } finally {
      setStoring(false);
    }
  };

  const handleDelete = (item: MemoryItem) => {
    Alert.alert('Delete Memory', `Delete this memory?`, [
      {text: 'Cancel', style: 'cancel'},
      {
        text: 'Delete',
        style: 'destructive',
        onPress: async () => {
          try {
            triggerHaptic('light');
            await api.delete(`/memory/${item.id}`);
            triggerHaptic('success');
            toast.success('Memory deleted');
            await fetchData();
          } catch {
            toast.error('Failed to delete memory');
          }
        },
      },
    ]);
  };

  const handleConsolidate = async () => {
    try {
      triggerHaptic('light');
      await api.post('/memory/consolidate');
      triggerHaptic('success');
      toast.success('Memory consolidated');
      await fetchData();
    } catch {
      toast.error('Consolidation failed');
    }
  };

  const accent = colors.primary;

  const formatTime = (ts: string) => {
    const d = new Date(ts);
    const now = new Date();
    const diff = now.getTime() - d.getTime();
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
    return d.toLocaleDateString();
  };

  const getImportanceColor = (importance: number) => {
    if (importance >= 0.8) return colors.error;
    if (importance >= 0.5) return colors.warning;
    return colors.success;
  };

  return (
    <SafeAreaView style={{flex: 1, backgroundColor: colors.background}} edges={['top']}>
      <XStack paddingHorizontal={16} paddingVertical={12} alignItems="center" justifyContent="space-between">
        <YStack>
          <Text fontSize={20} fontWeight="600" color={colors.text}>Memory</Text>
          <Text fontSize={12} color={colors.textSecondary}>
            {stats ? `${stats.total_items} items, ${stats.topics?.length ?? 0} topics` : `${items.length} items`}
          </Text>
        </YStack>
        <XStack gap={8}>
          <Pressable onPress={handleConsolidate} style={{padding: 8}}>
            <Icon name="layers" size={20} color={accent} />
          </Pressable>
          <Pressable onPress={onRefresh} style={{padding: 8}}>
            <Icon name="refresh-cw" size={20} color={accent} />
          </Pressable>
        </XStack>
      </XStack>

      {/* Tabs */}
      <XStack paddingHorizontal={16} gap={4} marginBottom={12}>
        {(['all', 'store'] as Tab[]).map(t => (
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
              backgroundColor: tab === t ? accent : colors.backgroundHover,
            }}>
            <Text
              fontSize={13}
              fontWeight={tab === t ? '600' : '400'}
              color={tab === t ? '#fff' : colors.text}>
              {t === 'all' ? 'All Memories' : 'Store New'}
            </Text>
          </Pressable>
        ))}
      </XStack>

      {/* Stats Row */}
      {stats && (
        <XStack paddingHorizontal={16} gap={8} marginBottom={12}>
          <YStack flex={1} backgroundColor={colors.backgroundHover} borderRadius={8} padding={8} alignItems="center">
            <Text fontSize={16} fontWeight="700" color={accent}>{stats.total_items}</Text>
            <Text fontSize={10} color={colors.textSecondary}>Items</Text>
          </YStack>
          <YStack flex={1} backgroundColor={colors.backgroundHover} borderRadius={8} padding={8} alignItems="center">
            <Text fontSize={16} fontWeight="700" color={colors.success}>{stats.topics?.length ?? 0}</Text>
            <Text fontSize={10} color={colors.textSecondary}>Topics</Text>
          </YStack>
          <YStack flex={1} backgroundColor={colors.backgroundHover} borderRadius={8} padding={8} alignItems="center">
            <StatusBadge
              label={stats.enabled ? 'Enabled' : 'Disabled'}
              variant={stats.enabled ? 'success' : 'warning'}
            />
          </YStack>
        </XStack>
      )}

      {tab === 'store' ? (
        /* Store Form */
        <YStack paddingHorizontal={16} flex={1}>
          <YStack backgroundColor={colors.backgroundHover} borderRadius={8} padding={12}>
            <Text fontSize={13} fontWeight="500" color={colors.text} marginBottom={8}>New Memory</Text>
            <RNTextInput
              value={storeContent}
              onChangeText={setStoreContent}
              placeholder="What should the AI remember?"
              placeholderTextColor={colors.textMuted}
              multiline
              numberOfLines={4}
              style={{
                backgroundColor: colors.background,
                borderRadius: 8,
                padding: 12,
                fontSize: 14,
                color: colors.text,
                textAlignVertical: 'top',
                minHeight: 100,
                marginBottom: 8,
              }}
            />
            <RNTextInput
              value={storeTopic}
              onChangeText={setStoreTopic}
              placeholder="Topic (optional)"
              placeholderTextColor={colors.textMuted}
              style={{
                backgroundColor: colors.background,
                borderRadius: 8,
                paddingHorizontal: 12,
                paddingVertical: 8,
                fontSize: 14,
                color: colors.text,
                marginBottom: 8,
              }}
            />
            <Pressable
              onPress={handleStore}
              disabled={storing || !storeContent.trim()}
              style={{
                backgroundColor: storing || !storeContent.trim() ? colors.textMuted : accent,
                borderRadius: 8,
                paddingVertical: 10,
                alignItems: 'center',
              }}>
              <Text fontSize={13} fontWeight="600" color="#fff">
                {storing ? 'Storing...' : 'Store Memory'}
              </Text>
            </Pressable>
          </YStack>
        </YStack>
      ) : (
        /* Memory List */
        <>
          {/* Search */}
          <YStack paddingHorizontal={16} marginBottom={12}>
            <XStack gap={8}>
              <RNTextInput
                value={searchQuery}
                onChangeText={setSearchQuery}
                onSubmitEditing={handleSearch}
                placeholder="Search memories..."
                placeholderTextColor={colors.textMuted}
                returnKeyType="search"
                style={{
                  flex: 1,
                  backgroundColor: colors.backgroundHover,
                  borderRadius: 8,
                  paddingHorizontal: 12,
                  paddingVertical: 8,
                  fontSize: 14,
                  color: colors.text,
                }}
              />
              <Pressable
                onPress={handleSearch}
                disabled={searching}
                style={{
                  backgroundColor: accent,
                  borderRadius: 8,
                  paddingHorizontal: 12,
                  paddingVertical: 8,
                  alignItems: 'center',
                  justifyContent: 'center',
                }}>
                <Icon name="search" size={18} color="#fff" />
              </Pressable>
            </XStack>
          </YStack>

          {loading ? (
            <YStack flex={1} alignItems="center" justifyContent="center">
              <Icon name="refresh-cw" size={24} color={colors.textSecondary} />
              <Text fontSize={13} color={colors.textSecondary} marginTop={8}>Loading memories...</Text>
            </YStack>
          ) : (
            <FlatList
              data={items}
              keyExtractor={item => item.id}
              contentContainerStyle={{paddingHorizontal: 16, paddingBottom: 20}}
              refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={accent} />}
              ListEmptyComponent={
                <YStack alignItems="center" paddingVertical={40}>
                  <Icon name="brain" size={32} color={colors.textSecondary} />
                  <Text fontSize={14} color={colors.textSecondary} marginTop={8}>No memories yet</Text>
                  <Text fontSize={12} color={colors.textMuted} marginTop={4}>Store one above or let the AI learn from conversations</Text>
                </YStack>
              }
              renderItem={({item}) => (
                <YStack
                  backgroundColor={colors.backgroundHover}
                  borderRadius={8}
                  padding={12}
                  marginBottom={8}
                  borderLeftWidth={3}
                  borderLeftColor={getImportanceColor(item.importance)}>
                  <XStack justifyContent="space-between" alignItems="flex-start">
                    <Text fontSize={13} color={colors.text} flex={1} numberOfLines={3}>
                      {item.content}
                    </Text>
                    <Pressable
                      onPress={() => handleDelete(item)}
                      style={{padding: 4}}>
                      <Icon name="trash-2" size={14} color={colors.error} />
                    </Pressable>
                  </XStack>
                  <XStack gap={6} marginTop={6}>
                    <StatusBadge label={item.topic} variant="info" />
                    <Text fontSize={11} color={colors.textMuted}>{formatTime(item.created_at)}</Text>
                    {item.importance > 0 && (
                      <Text fontSize={11} color={getImportanceColor(item.importance)}>
                        {(item.importance * 100).toFixed(0)}%
                      </Text>
                    )}
                  </XStack>
                </YStack>
              )}
            />
          )}
        </>
      )}
    </SafeAreaView>
  );
}
