import React, {useState, useCallback, useRef, useEffect} from 'react';
import {
  View,
  Text,
  TextInput,
  FlatList,
  TouchableOpacity,
  Modal,
  StyleSheet,
  ActivityIndicator,
  Keyboard,
} from 'react-native';
import {api} from '../services/api-client';
import {triggerHaptic} from '../services/haptics';
import {colors, spacing, radii, typography} from '../theme';
import type {SearchResult, SearchMatch} from '../types';

interface Props {
  visible: boolean;
  onClose: () => void;
  onSelectSession: (sessionId: string) => void;
}

export function SearchSessionsModal({visible, onClose, onSelectSession}: Props) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState(false);
  const inputRef = useRef<TextInput>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (visible) {
      setTimeout(() => inputRef.current?.focus(), 100);
      setQuery('');
      setResults([]);
      setSearched(false);
    }
  }, [visible]);

  const doSearch = useCallback(async (q: string) => {
    if (!q.trim()) {
      setResults([]);
      setSearched(false);
      return;
    }
    setLoading(true);
    setSearched(true);
    try {
      const data = await api.searchSessions(q, 20);
      setResults(data.results || []);
    } catch {
      setResults([]);
    } finally {
      setLoading(false);
    }
  }, []);

  const onChangeText = useCallback(
    (text: string) => {
      setQuery(text);
      if (debounceRef.current) clearTimeout(debounceRef.current);
      debounceRef.current = setTimeout(() => doSearch(text), 300);
    },
    [doSearch],
  );

  const handleSelect = useCallback(
    (sessionId: string) => {
      triggerHaptic('light');
      Keyboard.dismiss();
      onSelectSession(sessionId);
      onClose();
    },
    [onSelectSession, onClose],
  );

  const renderMatch = (match: SearchMatch, idx: number) => (
    <View key={idx} style={styles.matchRow}>
      <Text style={styles.matchRole}>
        {match.role === 'user' ? 'You' : 'Assistant'}:
      </Text>
      <Text style={styles.matchContent} numberOfLines={2}>
        {match.content}
      </Text>
    </View>
  );

  return (
    <Modal visible={visible} animationType="slide" transparent onRequestClose={onClose}>
      <View style={styles.overlay}>
        <View style={styles.sheet}>
          <View style={styles.header}>
            <Text style={styles.title}>Search Conversations</Text>
            <TouchableOpacity onPress={onClose}>
              <Text style={styles.closeBtn}>×</Text>
            </TouchableOpacity>
          </View>

          <TextInput
            ref={inputRef}
            style={styles.input}
            value={query}
            onChangeText={onChangeText}
            placeholder="Search all conversations..."
            placeholderTextColor={colors.textMuted}
            returnKeyType="search"
            autoCapitalize="none"
            autoCorrect={false}
          />

          {loading && (
            <ActivityIndicator
              style={styles.loader}
              color={colors.primary}
              size="small"
            />
          )}

          {searched && !loading && results.length === 0 && (
            <View style={styles.empty}>
              <Text style={styles.emptyText}>
                {query.trim() ? 'No results found' : 'Type to search'}
              </Text>
            </View>
          )}

          <FlatList
            data={results}
            keyExtractor={item => item.id}
            renderItem={({item}) => (
              <TouchableOpacity
                style={styles.resultItem}
                onPress={() => handleSelect(item.id)}
                activeOpacity={0.7}>
                <View style={styles.resultHeader}>
                  <Text style={styles.resultName} numberOfLines={1}>
                    {item.name || 'Untitled'}
                  </Text>
                  <Text style={styles.resultCount}>
                    {item.match_count} {item.match_count === 1 ? 'match' : 'matches'}
                  </Text>
                </View>
                {item.matches.map(renderMatch)}
              </TouchableOpacity>
            )}
            keyboardShouldPersistTaps="handled"
            contentContainerStyle={results.length === 0 ? styles.emptyList : undefined}
          />
        </View>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  overlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.4)',
    justifyContent: 'flex-end',
  },
  sheet: {
    backgroundColor: colors.background,
    borderTopLeftRadius: radii.lg,
    borderTopRightRadius: radii.lg,
    maxHeight: '85%',
    minHeight: '40%',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  title: {
    ...typography.h3,
    color: colors.text,
  },
  closeBtn: {
    fontSize: 24,
    color: colors.textMuted,
    padding: spacing.xs,
  },
  input: {
    margin: spacing.lg,
    backgroundColor: colors.surface,
    borderRadius: radii.md,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    ...typography.body,
    color: colors.text,
    borderWidth: 1,
    borderColor: colors.border,
  },
  loader: {
    marginVertical: spacing.md,
  },
  empty: {
    alignItems: 'center',
    paddingVertical: spacing.xxxl,
  },
  emptyText: {
    ...typography.body,
    color: colors.textMuted,
  },
  emptyList: {
    flex: 1,
  },
  resultItem: {
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  resultHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: spacing.xs,
  },
  resultName: {
    ...typography.body,
    color: colors.text,
    fontWeight: '600',
    flex: 1,
    marginRight: spacing.sm,
  },
  resultCount: {
    ...typography.small,
    color: colors.primary,
    fontWeight: '500',
  },
  matchRow: {
    flexDirection: 'row',
    gap: spacing.sm,
    paddingVertical: 2,
  },
  matchRole: {
    ...typography.small,
    color: colors.primary,
    fontWeight: '600',
    minWidth: 60,
  },
  matchContent: {
    ...typography.small,
    color: colors.textSecondary,
    flex: 1,
  },
});
