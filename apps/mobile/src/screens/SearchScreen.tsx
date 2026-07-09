import React, {useState, useCallback} from 'react';
import {FlatList, TextInput, Keyboard} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {YStack, XStack, Text} from 'tamagui';
import {useChatStore} from '../stores/chat-store';
import {api} from '../services/api-client';
import {Icon} from '../components/Icon';

interface SearchResult {
  sessionId: string;
  sessionTitle: string;
  role: string;
  content: string;
  timestamp: string;
}

export function SearchScreen() {
  const {loadSession} = useChatStore();
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
    try {
      const res = await api.get(`/chat/sessions/search?q=${encodeURIComponent(text)}&limit=20`) as {
        results: Array<{
          id: string;
          name: string;
          matches: Array<{role: string; content: string; timestamp: string}>;
        }>;
      };
      const found: SearchResult[] = [];
      for (const session of res.results) {
        for (const match of session.matches) {
          found.push({
            sessionId: session.id,
            sessionTitle: session.name,
            role: match.role,
            content: match.content,
            timestamp: match.timestamp,
          });
        }
      }
      setResults(found);
    } catch {
      setResults([]);
    }
    setSearching(false);
  }, []);

  const handleResultPress = async (result: SearchResult) => {
    Keyboard.dismiss();
    await loadSession(result.sessionId);
  };

  return (
    <SafeAreaView style={{flex: 1, backgroundColor: '#F5F0FF'}} edges={['top']}>
      <YStack flex={1}>
        <YStack paddingHorizontal={16} paddingTop={12} paddingBottom={8}>
          <Text fontSize={20} fontWeight="600" letterSpacing={-0.2} color="$color">
            Search Messages
          </Text>
        </YStack>

        <XStack
          marginHorizontal={16}
          backgroundColor="white"
          borderRadius={8}
          paddingHorizontal={12}
          borderWidth={1}
          borderColor="$borderColor"
          alignItems="center">
          <Icon name="search" size={16} color="#9B95A8" />
          <TextInput
            style={{flex: 1, fontSize: 14, color: '#1A1625', paddingVertical: 10, marginLeft: 8, }}
            value={query}
            onChangeText={handleSearch}
            placeholder="Search across all conversations..."
            placeholderTextColor="#9B95A8"
            autoFocus
            returnKeyType="search"
          />
          {query.length > 0 && (
            <YStack onPress={() => handleSearch('')}>
              <Icon name="x" size={16} color="#9B95A8" />
            </YStack>
          )}
        </XStack>

        {searching && (
          <Text fontSize={14} color="$color10" textAlign="center" paddingVertical={24}>
            Searching...
          </Text>
        )}

        {!searching && query.length >= 2 && results.length === 0 && (
          <Text fontSize={14} color="$color10" textAlign="center" paddingVertical={24}>
            No results found
          </Text>
        )}

        <FlatList
          data={results}
          keyExtractor={(item, i) => `${item.sessionId}-${item.role}-${i}`}
          keyboardDismissMode="on-drag"
          contentContainerStyle={{paddingHorizontal: 16, paddingTop: 12, gap: 8}}
          renderItem={({item}) => (
            <YStack
              backgroundColor="white"
              borderRadius={8}
              padding={12}
              borderWidth={1}
              borderColor="$borderColor"
              onPress={() => handleResultPress(item)}>
              <XStack justifyContent="space-between" alignItems="center" marginBottom={4}>
                <Text fontSize={11} fontWeight="600" color="$color9" flex={1} numberOfLines={1}>
                  {item.sessionTitle || 'Untitled'}
                </Text>
                <Text fontSize={11} fontWeight="500" color="$color10" marginLeft={8}>
                  {item.role === 'user' ? 'You' : 'AI'}
                </Text>
              </XStack>
              <Text fontSize={14} color="$color" lineHeight={20} numberOfLines={3}>
                {item.content}
              </Text>
              <Text fontSize={11} fontWeight="500" color="$color10" marginTop={4}>
                {item.timestamp ? new Date(item.timestamp).toLocaleDateString() : ''}
              </Text>
            </YStack>
          )}
          ListEmptyComponent={
            query.length < 2 ? (
              <YStack paddingHorizontal={32} paddingTop={64}>
                <Text fontSize={20} fontWeight="600" color="$color" marginBottom={12} textAlign="center">
                  Search Tips
                </Text>
                <Text fontSize={14} color="$color11" marginBottom={8} textAlign="center">
                  • Type at least 2 characters
                </Text>
                <Text fontSize={14} color="$color11" marginBottom={8} textAlign="center">
                  • Searches across all your conversations
                </Text>
                <Text fontSize={14} color="$color11" textAlign="center">
                  • Tap a result to jump to that conversation
                </Text>
              </YStack>
            ) : null
          }
        />
      </YStack>
    </SafeAreaView>
  );
}
