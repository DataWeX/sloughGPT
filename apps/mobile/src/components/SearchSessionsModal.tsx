import React, {useState, useCallback, useRef, useEffect} from 'react';
import {
  TextInput,
  FlatList,
  Pressable,
  Modal,
  ActivityIndicator,
  Keyboard,
} from 'react-native';
import {YStack, XStack, Text, useTheme} from 'tamagui';
import {api} from '../services/api-client';
import {triggerHaptic} from '../services/haptics';
import {Icon} from '../components/Icon';
import type {SearchResult, SearchMatch} from '../types';

interface Props {
  visible: boolean;
  onClose: () => void;
  onSelectSession: (sessionId: string) => void;
}

export function SearchSessionsModal({visible, onClose, onSelectSession}: Props) {
  const theme = useTheme();
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

  const accent = theme.color9?.val || '#7C52C4';
  const mutedColor = theme.color10?.val || '#9B95A8';

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
      setResults(Array.isArray(data) ? data : []);
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
    <XStack key={idx} gap={8} paddingVertical={2}>
      <Text fontSize={11} fontWeight="600" letterSpacing={0.2} color="$color9" minWidth={60}>
        {match.role === 'user' ? 'You' : 'Assistant'}:
      </Text>
      <Text fontSize={11} fontWeight="500" letterSpacing={0.2} color="$color11" flex={1} numberOfLines={2}>
        {match.content}
      </Text>
    </XStack>
  );

  return (
    <Modal visible={visible} animationType="slide" transparent onRequestClose={onClose}>
      <YStack flex={1} backgroundColor="rgba(0,0,0,0.4)" justifyContent="flex-end">
        <YStack
          backgroundColor="$background"
          borderTopLeftRadius={24}
          borderTopRightRadius={24}
          maxHeight="85%"
          minHeight="40%">
          <XStack
            alignItems="center"
            justifyContent="space-between"
            paddingHorizontal={20}
            paddingVertical={14}
            borderBottomWidth={0.5}
            borderBottomColor="$borderColor">
            <Text fontSize={17} fontWeight="700" letterSpacing={-0.3} color="$color">
              Search Conversations
            </Text>
            <Pressable onPress={onClose}>
              <YStack width={28} height={28} borderRadius={9} alignItems="center" justifyContent="center">
                <Icon name="x" size={16} color={(theme.color11?.val || '#6B7280')} />
              </YStack>
            </Pressable>
          </XStack>

          <YStack marginHorizontal={16} marginTop={12} marginBottom={8}>
            <TextInput
              ref={inputRef}
              value={query}
              onChangeText={onChangeText}
              placeholder="Search all conversations..."
              placeholderTextColor={mutedColor}
              returnKeyType="search"
              autoCapitalize="none"
              autoCorrect={false}
              style={{
                backgroundColor: 'rgba(124, 82, 196, 0.04)',
                borderRadius: 10,
                paddingHorizontal: 14,
                paddingVertical: 10,
                fontSize: 14,
                color: '#1A1625',
                borderWidth: 0.5,
                borderColor: 'rgba(124, 82, 196, 0.12)',
              }}
            />
          </YStack>

          {loading && (
            <ActivityIndicator style={{marginVertical: 12}} color={accent} size="small" />
          )}

          {searched && !loading && results.length === 0 && (
            <YStack alignItems="center" paddingVertical={48}>
              <Text fontSize={15} fontWeight="400" color="$color10">
                {query.trim() ? 'No results found' : 'Type to search'}
              </Text>
            </YStack>
          )}

          <FlatList
            data={results}
            keyExtractor={item => item.id}
            renderItem={({item}) => (
              <Pressable onPress={() => handleSelect(item.id)}>
                <YStack paddingHorizontal={16} paddingVertical={12} borderBottomWidth={0.5} borderBottomColor="$borderColor">
                  <XStack alignItems="center" justifyContent="space-between" marginBottom={4}>
                    <Text fontSize={15} fontWeight="600" color="$color" flex={1} marginRight={8} numberOfLines={1}>
                      {item.name || 'Untitled'}
                    </Text>
                    <Text fontSize={11} fontWeight="500" letterSpacing={0.2} color="$color9">
                      {item.match_count} {item.match_count === 1 ? 'match' : 'matches'}
                    </Text>
                  </XStack>
                  {item.matches.map(renderMatch)}
                </YStack>
              </Pressable>
            )}
            keyboardShouldPersistTaps="handled"
            contentContainerStyle={results.length === 0 ? {flex: 1} : undefined}
          />
        </YStack>
      </YStack>
    </Modal>
  );
}
