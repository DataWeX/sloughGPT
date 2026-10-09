import React, {useEffect, useState, useCallback} from 'react';
import {FlatList, Pressable, RefreshControl, TextInput as RNTextInput} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {YStack, XStack, Text} from 'tamagui';
import {useColors} from '../theme/colors';
import {api} from '../services/api-client';
import {Icon} from '../components/Icon';
import {StatusBadge} from '../components/StatusBadge';
import {triggerHaptic} from '../services/haptics';
import {toast} from '../services/toast';

type Tab = 'search' | 'ingest' | 'knowledge' | 'feeds';

interface LearnStatus {
  knowledge_count: number;
  tokens: number;
  feeds: number;
  status: string;
}

interface KnowledgeFact {
  id: string;
  content: string;
  topic: string | null;
  importance: number;
}

interface Feed {
  id: string;
  url: string;
  interval: number;
  last_fetched: string | null;
}

export function LearnScreen() {
  const colors = useColors();
  const [tab, setTab] = useState<Tab>('search');
  const [status, setStatus] = useState<LearnStatus | null>(null);
  const [facts, setFacts] = useState<KnowledgeFact[]>([]);
  const [feeds, setFeeds] = useState<Feed[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  // Search
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<KnowledgeFact[]>([]);
  const [searching, setSearching] = useState(false);

  // Ingest
  const [ingestUrl, setIngestUrl] = useState('');
  const [ingestText, setIngestText] = useState('');
  const [ingesting, setIngesting] = useState(false);

  // Feeds
  const [feedUrl, setFeedUrl] = useState('');
  const [addingFeed, setAddingFeed] = useState(false);

  const fetchAll = useCallback(async () => {
    try {
      const [s, k, f] = await Promise.all([
        api.get<LearnStatus>('/learn/status').catch(() => null),
        api.get<{facts: KnowledgeFact[]}>('/learn/knowledge?limit=50').catch(() => ({facts: []})),
        api.get<{feeds: Feed[]}>('/learn/feed?action=list').catch(() => ({feeds: []})),
      ]);
      setStatus(s);
      setFacts(k.facts || []);
      setFeeds(f.feeds || []);
    } catch {
      // handled above
    }
  }, []);

  useEffect(() => {
    fetchAll().finally(() => setLoading(false));
  }, [fetchAll]);

  const onRefresh = async () => {
    setRefreshing(true);
    await fetchAll();
    setRefreshing(false);
  };

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    try {
      setSearching(true);
      const result = await api.post<{results: KnowledgeFact[]}>('/learn/search', {query: searchQuery.trim()});
      setSearchResults(result.results || []);
    } catch {
      toast.error('Search failed');
    } finally {
      setSearching(false);
    }
  };

  const handleIngestUrl = async () => {
    if (!ingestUrl.trim()) return;
    try {
      setIngesting(true);
      triggerHaptic('light');
      await api.post('/learn/ingest-url', {url: ingestUrl.trim()});
      triggerHaptic('success');
      toast.success('URL ingested');
      setIngestUrl('');
      await fetchAll();
    } catch {
      toast.error('Ingest failed');
    } finally {
      setIngesting(false);
    }
  };

  const handleIngestText = async () => {
    if (!ingestText.trim()) return;
    try {
      setIngesting(true);
      triggerHaptic('light');
      await api.post('/learn/ingest', {text: ingestText.trim()});
      triggerHaptic('success');
      toast.success('Text ingested');
      setIngestText('');
      await fetchAll();
    } catch {
      toast.error('Ingest failed');
    } finally {
      setIngesting(false);
    }
  };

  const handleAddFeed = async () => {
    if (!feedUrl.trim()) return;
    try {
      setAddingFeed(true);
      triggerHaptic('light');
      await api.post('/learn/feed', {url: feedUrl.trim()});
      triggerHaptic('success');
      toast.success('Feed added');
      setFeedUrl('');
      await fetchAll();
    } catch {
      toast.error('Failed to add feed');
    } finally {
      setAddingFeed(false);
    }
  };

  const TABS: {key: Tab; label: string; icon: string}[] = [
    {key: 'search', label: 'Search', icon: 'search'},
    {key: 'ingest', label: 'Ingest', icon: 'upload'},
    {key: 'knowledge', label: 'Knowledge', icon: 'book-open'},
    {key: 'feeds', label: 'Feeds', icon: 'bookmark'},
  ];

  const renderFact = ({item}: {item: KnowledgeFact}) => (
    <YStack padding={10} borderRadius={8} backgroundColor={colors.background} gap={4}>
      <Text fontSize={13} color={colors.text} numberOfLines={3}>{item.content}</Text>
      <XStack gap={6}>
        {item.topic && <StatusBadge label={item.topic} variant="info" />}
        <StatusBadge label={`importance: ${item.importance.toFixed(1)}`} variant="default" />
      </XStack>
    </YStack>
  );

  const renderFeed = ({item}: {item: Feed}) => (
    <XStack padding={10} borderRadius={8} backgroundColor={colors.background} gap={8} alignItems="center">
      <Icon name="bookmark" size={16} color={colors.primary} />
      <YStack flex={1} gap={2}>
        <Text fontSize={13} color={colors.text} numberOfLines={1}>{item.url}</Text>
        <Text fontSize={11} color={colors.textMuted}>Every {item.interval}s</Text>
      </YStack>
    </XStack>
  );

  const renderTabContent = () => {
    switch (tab) {
      case 'search':
        return (
          <YStack gap={8}>
            <XStack gap={8}>
              <RNTextInput
                value={searchQuery}
                onChangeText={setSearchQuery}
                placeholder="Search knowledge..."
                placeholderTextColor={colors.textMuted}
                onSubmitEditing={handleSearch}
                style={{
                  flex: 1,
                  borderWidth: 1,
                  borderColor: colors.border,
                  borderRadius: 8,
                  padding: 10,
                  fontSize: 14,
                  color: colors.text,
                  backgroundColor: colors.white,
                }}
              />
              <Pressable onPress={handleSearch} disabled={!searchQuery.trim() || searching}>
                <XStack paddingHorizontal={12} paddingVertical={10} borderRadius={8} backgroundColor={colors.primary} alignItems="center">
                  <Icon name="search" size={16} color="white" />
                </XStack>
              </Pressable>
            </XStack>
            {searchResults.length > 0 && (
              <Text fontSize={12} color={colors.textMuted}>{searchResults.length} results found</Text>
            )}
            <FlatList data={searchResults} keyExtractor={item => item.id} renderItem={renderFact} scrollEnabled={false} />
          </YStack>
        );
      case 'ingest':
        return (
          <YStack gap={12}>
            <YStack gap={6}>
              <Text fontSize={13} fontWeight="500" color={colors.text}>Ingest from URL</Text>
              <RNTextInput
                value={ingestUrl}
                onChangeText={setIngestUrl}
                placeholder="https://example.com/article"
                placeholderTextColor={colors.textMuted}
                style={{
                  borderWidth: 1,
                  borderColor: colors.border,
                  borderRadius: 8,
                  padding: 10,
                  fontSize: 14,
                  color: colors.text,
                  backgroundColor: colors.white,
                }}
              />
              <Pressable onPress={handleIngestUrl} disabled={!ingestUrl.trim() || ingesting}>
                <XStack padding={10} borderRadius={8} backgroundColor={ingestUrl.trim() && !ingesting ? colors.primary : colors.border} alignItems="center" justifyContent="center" gap={6}>
                  <Icon name="upload" size={16} color="white" />
                  <Text fontSize={13} fontWeight="600" color="white">{ingesting ? 'Ingesting...' : 'Ingest URL'}</Text>
                </XStack>
              </Pressable>
            </YStack>
            <YStack gap={6}>
              <Text fontSize={13} fontWeight="500" color={colors.text}>Ingest Text</Text>
              <RNTextInput
                value={ingestText}
                onChangeText={setIngestText}
                placeholder="Paste text to ingest..."
                placeholderTextColor={colors.textMuted}
                multiline
                numberOfLines={4}
                style={{
                  borderWidth: 1,
                  borderColor: colors.border,
                  borderRadius: 8,
                  padding: 10,
                  fontSize: 14,
                  color: colors.text,
                  backgroundColor: colors.white,
                  minHeight: 100,
                  textAlignVertical: 'top',
                }}
              />
              <Pressable onPress={handleIngestText} disabled={!ingestText.trim() || ingesting}>
                <XStack padding={10} borderRadius={8} backgroundColor={ingestText.trim() && !ingesting ? colors.primary : colors.border} alignItems="center" justifyContent="center" gap={6}>
                  <Icon name="upload" size={16} color="white" />
                  <Text fontSize={13} fontWeight="600" color="white">{ingesting ? 'Ingesting...' : 'Ingest Text'}</Text>
                </XStack>
              </Pressable>
            </YStack>
          </YStack>
        );
      case 'knowledge':
        return (
          <YStack gap={6}>
            {facts.length === 0 ? (
              <YStack padding={20} alignItems="center" gap={8}>
                <Icon name="book-open" size={24} color={colors.textMuted} />
                <Text fontSize={13} color={colors.textMuted}>No knowledge facts yet</Text>
              </YStack>
            ) : (
              facts.map(f => (
                <YStack key={f.id} padding={10} borderRadius={8} backgroundColor={colors.background} gap={4}>
                  <Text fontSize={13} color={colors.text} numberOfLines={2}>{f.content}</Text>
                  {f.topic && <StatusBadge label={f.topic} variant="info" />}
                </YStack>
              ))
            )}
          </YStack>
        );
      case 'feeds':
        return (
          <YStack gap={8}>
            <XStack gap={8}>
              <RNTextInput
                value={feedUrl}
                onChangeText={setFeedUrl}
                placeholder="RSS feed URL..."
                placeholderTextColor={colors.textMuted}
                style={{
                  flex: 1,
                  borderWidth: 1,
                  borderColor: colors.border,
                  borderRadius: 8,
                  padding: 10,
                  fontSize: 14,
                  color: colors.text,
                  backgroundColor: colors.white,
                }}
              />
              <Pressable onPress={handleAddFeed} disabled={!feedUrl.trim() || addingFeed}>
                <XStack paddingHorizontal={12} paddingVertical={10} borderRadius={8} backgroundColor={colors.primary} alignItems="center">
                  <Icon name="plus" size={16} color="white" />
                </XStack>
              </Pressable>
            </XStack>
            {feeds.length === 0 ? (
              <YStack padding={20} alignItems="center" gap={8}>
                <Icon name="bookmark" size={24} color={colors.textMuted} />
                <Text fontSize={13} color={colors.textMuted}>No feeds subscribed</Text>
              </YStack>
            ) : (
              <FlatList data={feeds} keyExtractor={item => item.id} renderItem={renderFeed} scrollEnabled={false} />
            )}
          </YStack>
        );
    }
  };

  return (
    <SafeAreaView style={{flex: 1, backgroundColor: colors.background}} edges={['top']}>
      <XStack paddingHorizontal={16} paddingVertical={12} alignItems="center" justifyContent="space-between">
        <Text fontSize={20} fontWeight="600" color={colors.text}>Learn</Text>
        <Pressable onPress={onRefresh}>
          <Icon name="refresh-cw" size={18} color={colors.primary} />
        </Pressable>
      </XStack>

      {loading ? (
        <YStack flex={1} alignItems="center" justifyContent="center">
          <StatusBadge label="Loading..." variant="info" />
        </YStack>
      ) : (
        <YStack flex={1} paddingHorizontal={16} gap={12}>
          {/* Stats */}
          <XStack gap={8}>
            <YStack flex={1} padding={10} borderRadius={8} backgroundColor={colors.card} borderWidth={0.5} borderColor={colors.border} gap={2} alignItems="center">
              <Text fontSize={18} fontWeight="700" color={colors.text}>{status?.knowledge_count ?? facts.length}</Text>
              <Text fontSize={10} color={colors.textMuted}>Facts</Text>
            </YStack>
            <YStack flex={1} padding={10} borderRadius={8} backgroundColor={colors.card} borderWidth={0.5} borderColor={colors.border} gap={2} alignItems="center">
              <Text fontSize={18} fontWeight="700" color={colors.text}>{status?.tokens ?? 0}</Text>
              <Text fontSize={10} color={colors.textMuted}>Tokens</Text>
            </YStack>
            <YStack flex={1} padding={10} borderRadius={8} backgroundColor={colors.card} borderWidth={0.5} borderColor={colors.border} gap={2} alignItems="center">
              <Text fontSize={18} fontWeight="700" color={colors.text}>{status?.feeds ?? feeds.length}</Text>
              <Text fontSize={10} color={colors.textMuted}>Feeds</Text>
            </YStack>
          </XStack>

          {/* Tabs */}
          <XStack gap={4}>
            {TABS.map(t => (
              <Pressable key={t.key} onPress={() => setTab(t.key)} style={{flex: 1}}>
                <XStack paddingVertical={6} borderRadius={6} backgroundColor={tab === t.key ? colors.primary : 'transparent'} alignItems="center" justifyContent="center" gap={4}>
                  <Icon name={t.icon as any} size={12} color={tab === t.key ? 'white' : colors.textMuted} />
                  <Text fontSize={11} fontWeight={tab === t.key ? '600' : '400'} color={tab === t.key ? 'white' : colors.textMuted}>{t.label}</Text>
                </XStack>
              </Pressable>
            ))}
          </XStack>

          {/* Tab Content */}
          <YStack flex={1}>
            {renderTabContent()}
          </YStack>
        </YStack>
      )}
    </SafeAreaView>
  );
}
