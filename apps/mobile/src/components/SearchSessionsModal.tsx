import React, {useState, useCallback, useRef, useEffect} from 'react';
import {
  TextInput,
  FlatList,
  TouchableOpacity,
  Modal,
  ActivityIndicator,
  Keyboard,
} from 'react-native';
import {YStack, XStack, Text} from 'tamagui';
import {api} from '../services/api-client';
import {triggerHaptic} from '../services/haptics';
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
          borderTopLeftRadius={12}
          borderTopRightRadius={12}
          maxHeight="85%"
          minHeight="40%">
          {/* Header */}
          <XStack
            alignItems="center"
            justifyContent="space-between"
            paddingHorizontal={16}
            paddingVertical={12}
            borderBottomWidth={1}
            borderBottomColor="$borderColor">
            <Text fontSize={18} fontWeight="600" letterSpacing={-0.2} color="$color">
              Search Conversations
            </Text>
            <TouchableOpacity onPress={onClose} style={{padding: 4}}>
              <Text fontSize={24} color="$color10">
                ×
              </Text>
            </TouchableOpacity>
          </XStack>

          {/* Search Input */}
          <TextInput
            ref={inputRef}
            value={query}
            onChangeText={onChangeText}
            placeholder="Search all conversations..."
            placeholderTextColor="$color10"
            returnKeyType="search"
            autoCapitalize="none"
            autoCorrect={false}
            style={{
              margin: 16,
              backgroundColor: 'white',
              borderRadius: 8,
              paddingHorizontal: 12,
              paddingVertical: 8,
              fontSize: 15,
              color: '$color',
              borderWidth: 1,
              borderColor: '$borderColor',
            }}
          />

          {loading && (
            <ActivityIndicator style={{marginVertical: 12}} color="$color9" size="small" />
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
              <TouchableOpacity
                onPress={() => handleSelect(item.id)}
                activeOpacity={0.7}
                style={{
                  paddingHorizontal: 16,
                  paddingVertical: 12,
                  borderBottomWidth: 1,
                  borderBottomColor: '$borderColor',
                }}>
                <XStack alignItems="center" justifyContent="space-between" marginBottom={4}>
                  <Text fontSize={15} fontWeight="600" color="$color" flex={1} marginRight={8} numberOfLines={1}>
                    {item.name || 'Untitled'}
                  </Text>
                  <Text fontSize={11} fontWeight="500" letterSpacing={0.2} color="$color9">
                    {item.match_count} {item.match_count === 1 ? 'match' : 'matches'}
                  </Text>
                </XStack>
                {item.matches.map(renderMatch)}
              </TouchableOpacity>
            )}
            keyboardShouldPersistTaps="handled"
            contentContainerStyle={results.length === 0 ? {flex: 1} : undefined}
          />
        </YStack>
      </YStack>
    </Modal>
  );
}
