import React, {useState, useCallback} from 'react';
import {
  View,
  Text,
  TextInput,
  FlatList,
  TouchableOpacity,
  StyleSheet,
  Keyboard,
} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {useChatStore} from '../stores/chat-store';
import {colors, spacing, radii, typography} from '../theme';
import type {Message} from '../types';

interface SearchResult {
  sessionId: string;
  sessionTitle: string;
  message: Message;
}

export function SearchScreen() {
  const {sessions, loadSession} = useChatStore();
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [searching, setSearching] = useState(false);

  const handleSearch = useCallback(async (text: string) => {
    setQuery(text);
    if (text.trim().length < 2) {
      setResults([]);
      return;
    }

    setSearching(true);
    const lower = text.toLowerCase();
    const found: SearchResult[] = [];

    // Search through cached sessions
    for (const session of sessions) {
      try {
        const {api} = require('../services/api-client');
        const data = await api.get(`/chat/sessions/${session.id}`) as {messages: Message[]};
        for (const msg of data.messages) {
          if (msg.content.toLowerCase().includes(lower)) {
            found.push({
              sessionId: session.id,
              sessionTitle: session.title,
              message: msg,
            });
          }
        }
      } catch {
        // skip failed sessions
      }
    }

    setResults(found);
    setSearching(false);
  }, [sessions]);

  const handleResultPress = async (result: SearchResult) => {
    Keyboard.dismiss();
    await loadSession(result.sessionId);
  };

  const highlightMatch = (text: string, q: string) => {
    if (!q.trim()) return text;
    const parts = text.split(new RegExp(`(${q.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi'));
    return parts;
  };

  return (
    <SafeAreaView style={styles.safe} edges={['top']}>
      <View style={styles.container}>
        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.title}>Search Messages</Text>
        </View>

        {/* Search bar */}
        <View style={styles.searchBar}>
          <Text style={styles.searchIcon}>🔍</Text>
          <TextInput
            style={styles.searchInput}
            value={query}
            onChangeText={handleSearch}
            placeholder="Search across all conversations..."
            placeholderTextColor={colors.textMuted}
            autoFocus
            returnKeyType="search"
          />
          {query.length > 0 && (
            <TouchableOpacity onPress={() => handleSearch('')}>
              <Text style={styles.clearBtn}>✕</Text>
            </TouchableOpacity>
          )}
        </View>

        {/* Results */}
        {searching && (
          <Text style={styles.status}>Searching...</Text>
        )}

        {!searching && query.length >= 2 && results.length === 0 && (
          <Text style={styles.status}>No results found</Text>
        )}

        <FlatList
          data={results}
          keyExtractor={(item, i) => `${item.sessionId}-${item.message.id}-${i}`}
          keyboardDismissMode="on-drag"
          contentContainerStyle={styles.list}
          renderItem={({item}) => (
            <TouchableOpacity
              style={styles.resultItem}
              onPress={() => handleResultPress(item)}
              activeOpacity={0.7}>
              <View style={styles.resultHeader}>
                <Text style={styles.sessionTitle} numberOfLines={1}>
                  {item.sessionTitle || 'Untitled'}
                </Text>
                <Text style={styles.role}>
                  {item.message.role === 'user' ? 'You' : 'AI'}
                </Text>
              </View>
              <Text style={styles.preview} numberOfLines={3}>
                {item.message.content}
              </Text>
              <Text style={styles.timestamp}>
                {new Date(item.message.timestamp).toLocaleDateString()}
              </Text>
            </TouchableOpacity>
          )}
        />

        {query.length < 2 && (
          <View style={styles.hints}>
            <Text style={styles.hintTitle}>Search Tips</Text>
            <Text style={styles.hint}>• Type at least 2 characters</Text>
            <Text style={styles.hint}>• Searches across all your conversations</Text>
            <Text style={styles.hint}>• Tap a result to jump to that conversation</Text>
          </View>
        )}
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: colors.background,
  },
  container: {
    flex: 1,
  },
  header: {
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.md,
    paddingBottom: spacing.sm,
  },
  title: {
    ...typography.h2,
    color: colors.text,
  },
  searchBar: {
    flexDirection: 'row',
    alignItems: 'center',
    marginHorizontal: spacing.lg,
    backgroundColor: colors.surface,
    borderRadius: radii.md,
    paddingHorizontal: spacing.md,
    borderWidth: 1,
    borderColor: colors.border,
  },
  searchIcon: {
    fontSize: 16,
    marginRight: spacing.sm,
  },
  searchInput: {
    flex: 1,
    ...typography.body,
    color: colors.text,
    paddingVertical: spacing.sm + 2,
  },
  clearBtn: {
    fontSize: 16,
    color: colors.textMuted,
    padding: spacing.xs,
  },
  status: {
    ...typography.body,
    color: colors.textMuted,
    textAlign: 'center',
    paddingVertical: spacing.xxl,
  },
  list: {
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.md,
    gap: spacing.sm,
  },
  resultItem: {
    backgroundColor: colors.surface,
    borderRadius: radii.md,
    padding: spacing.md,
    borderWidth: 1,
    borderColor: colors.border,
  },
  resultHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 4,
  },
  sessionTitle: {
    ...typography.small,
    color: colors.primary,
    fontWeight: '600',
    flex: 1,
  },
  role: {
    ...typography.small,
    color: colors.textMuted,
    marginLeft: spacing.sm,
  },
  preview: {
    ...typography.body,
    color: colors.text,
    lineHeight: 20,
  },
  timestamp: {
    ...typography.small,
    color: colors.textMuted,
    marginTop: 4,
  },
  hints: {
    paddingHorizontal: spacing.xxxl,
    paddingTop: spacing.xxxl * 2,
  },
  hintTitle: {
    ...typography.h2,
    color: colors.text,
    marginBottom: spacing.md,
    textAlign: 'center',
  },
  hint: {
    ...typography.body,
    color: colors.textSecondary,
    marginBottom: spacing.sm,
    textAlign: 'center',
  },
});
