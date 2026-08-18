import React, {useState, useEffect, useCallback} from 'react';
import {RefreshControl, Pressable, TextInput as RNTextInput, ScrollView, FlatList} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {YStack, XStack, Text} from 'tamagui';
import {useColors} from '../theme/colors';
import {api} from '../services/api-client';
import {StatusBadge} from '../components/StatusBadge';
import {Icon} from '../components/Icon';
import {toast} from '../services/toast';

interface TokenizerStats {
  vocab_size: number;
  total_merges: number;
  model_name: string | null;
}

interface SampleWord {
  word: string;
  tokens: string[];
  ids: number[];
  count: number;
}

interface TokenizeResult {
  tokens: string[];
  ids: number[];
}

export function TokenizerScreen() {
  const colors = useColors();

  const [stats, setStats] = useState<TokenizerStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Playground
  const [inputText, setInputText] = useState('');
  const [tokenResult, setTokenResult] = useState<TokenizeResult | null>(null);
  const [tokenizing, setTokenizing] = useState(false);

  // Samples
  const [samples, setSamples] = useState<SampleWord[]>([]);
  const [samplesLoaded, setSamplesLoaded] = useState(false);

  const fetchData = useCallback(async () => {
    try {
      const [s, samp] = await Promise.all([
        api.get<TokenizerStats>('/tokenizer/stats').catch(() => null),
        api.get<SampleWord[]>('/tokenizer/samples').catch(() => []),
      ]);
      setStats(s);
      setSamples(samp || []);
      setSamplesLoaded(true);
      setError(null);
    } catch {
      setError('Could not load tokenizer data');
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

  const handleTokenize = async () => {
    if (!inputText.trim()) return;
    setTokenizing(true);
    try {
      const result = await api.post<TokenizeResult>('/tokenizer/tokenize', {text: inputText});
      setTokenResult(result);
    } catch {
      toast.error('Tokenization failed');
    } finally {
      setTokenizing(false);
    }
  };

  const handleClear = () => {
    setInputText('');
    setTokenResult(null);
  };

  if (loading) {
    return (
      <SafeAreaView style={{flex: 1}} edges={['top']}>
        <YStack flex={1} backgroundColor={colors.background} alignItems="center" justifyContent="center">
          <StatusBadge label="Loading..." variant="info" />
        </YStack>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={{flex: 1}} edges={['top']}>
      <ScrollView
        style={{flex: 1, backgroundColor: colors.background}}
        contentContainerStyle={{padding: 16, gap: 12}}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}>
        <Text fontSize={24} fontWeight="700" letterSpacing={-0.3} color="$color" paddingBottom={4}>
          Tokenizer
        </Text>

        {error && (
          <YStack
            backgroundColor={colors.errorAlpha(0.06)}
            borderRadius={10}
            padding={12}
            borderWidth={0.5}
            borderColor={colors.errorAlpha(0.12)}>
            <Text fontSize={13} color={colors.error}>{error}</Text>
          </YStack>
        )}

        {/* Stats */}
        {stats && (
          <XStack gap={10}>
            <YStack
              flex={1}
              backgroundColor={colors.white}
              borderRadius={10}
              borderWidth={0.5}
              borderColor={colors.border}
              padding={14}
              alignItems="center"
              gap={4}>
              <Text fontSize={22} fontWeight="700" color="$color">
                {stats.vocab_size.toLocaleString()}
              </Text>
              <Text fontSize={11} fontWeight="500" color="$color10">Vocab Size</Text>
            </YStack>
            <YStack
              flex={1}
              backgroundColor={colors.white}
              borderRadius={10}
              borderWidth={0.5}
              borderColor={colors.border}
              padding={14}
              alignItems="center"
              gap={4}>
              <Text fontSize={22} fontWeight="700" color="$color">
                {stats.total_merges.toLocaleString()}
              </Text>
              <Text fontSize={11} fontWeight="500" color="$color10">Merges</Text>
            </YStack>
            {stats.model_name && (
              <YStack
                flex={1}
                backgroundColor={colors.white}
                borderRadius={10}
                borderWidth={0.5}
                borderColor={colors.border}
                padding={14}
                alignItems="center"
                gap={4}>
                <Text fontSize={13} fontWeight="600" color="$color" numberOfLines={1}>
                  {stats.model_name}
                </Text>
                <Text fontSize={11} fontWeight="500" color="$color10">Model</Text>
              </YStack>
            )}
          </XStack>
        )}

        {/* Playground */}
        <YStack
          backgroundColor={colors.white}
          borderRadius={12}
          borderWidth={0.5}
          borderColor={colors.border}
          padding={14}
          gap={10}>
          <XStack alignItems="center" gap={6}>
            <Icon name="book" size={16} color={colors.primary} />
            <Text fontSize={15} fontWeight="600" color="$color">Playground</Text>
          </XStack>

          <RNTextInput
            style={{
              fontSize: 14,
              color: colors.text,
              backgroundColor: colors.primaryAlpha(0.04),
              borderRadius: 8,
              borderWidth: 0.5,
              borderColor: colors.border,
              paddingHorizontal: 12,
              paddingVertical: 10,
              minHeight: 60,
              textAlignVertical: 'top',
            }}
            value={inputText}
            onChangeText={setInputText}
            placeholder="Enter text to tokenize..."
            placeholderTextColor={colors.textMuted}
            multiline
          />

          <XStack gap={8}>
            <Pressable
              onPress={handleTokenize}
              disabled={tokenizing || !inputText.trim()}
              style={{flex: 1}}>
              {({pressed}) => (
                <YStack
                  backgroundColor={pressed ? colors.primary + 'CC' : colors.primary}
                  borderRadius={8}
                  paddingVertical={9}
                  alignItems="center"
                  opacity={!inputText.trim() || tokenizing ? 0.5 : 1}>
                  <Text fontSize={13} fontWeight="600" color={colors.white}>
                    {tokenizing ? 'Tokenizing...' : 'Tokenize'}
                  </Text>
                </YStack>
              )}
            </Pressable>
            {inputText.length > 0 && (
              <Pressable onPress={handleClear}>
                {({pressed}) => (
                  <YStack
                    backgroundColor={pressed ? colors.errorAlpha(0.1) : 'transparent'}
                    borderRadius={8}
                    borderWidth={0.5}
                    borderColor={colors.border}
                    paddingHorizontal={14}
                    paddingVertical={9}
                    alignItems="center"
                    justifyContent="center">
                    <Icon name="x" size={14} color={colors.textMuted} />
                  </YStack>
                )}
              </Pressable>
            )}
          </XStack>

          {/* Results */}
          {tokenResult && (
            <YStack gap={6}>
              <XStack justifyContent="space-between" alignItems="center">
                <Text fontSize={11} fontWeight="600" color="$color10" letterSpacing={0.5}>
                  TOKENS ({tokenResult.ids.length})
                </Text>
                <Pressable
                  onPress={() => {
                    const text = tokenResult.tokens.join(' ');
                    const {Clipboard} = require('react-native');
                    Clipboard.setString(text);
                    toast.success('Tokens copied');
                  }}>
                  {({pressed}) => (
                    <XStack gap={4} alignItems="center" opacity={pressed ? 0.5 : 1}>
                      <Icon name="copy" size={12} color={colors.primary} />
                      <Text fontSize={11} color={colors.primary}>Copy</Text>
                    </XStack>
                  )}
                </Pressable>
              </XStack>
              <YStack
                backgroundColor={colors.primaryAlpha(0.04)}
                borderRadius={8}
                padding={10}
                borderWidth={0.5}
                borderColor={colors.primaryAlpha(0.1)}>
                <Text fontSize={12} fontFamily="monospace" color="$color" selectable>
                  {tokenResult.tokens.map((t, i) =>
                    i === 0 ? t : ` · ${t}`,
                  ).join('')}
                </Text>
              </YStack>
              <YStack
                backgroundColor={colors.successAlpha(0.04)}
                borderRadius={8}
                padding={10}
                borderWidth={0.5}
                borderColor={colors.successAlpha(0.1)}>
                <Text fontSize={11} fontWeight="600" color={colors.success} marginBottom={2}>
                  IDs
                </Text>
                <Text fontSize={11} fontFamily="monospace" color="$color11" selectable>
                  [{tokenResult.ids.join(', ')}]
                </Text>
              </YStack>
            </YStack>
          )}
        </YStack>

        {/* Sample Words */}
        {samplesLoaded && samples.length > 0 && (
          <YStack
            backgroundColor={colors.white}
            borderRadius={12}
            borderWidth={0.5}
            borderColor={colors.border}
            padding={14}
            gap={10}>
            <XStack alignItems="center" gap={6}>
              <Icon name="book-open" size={16} color={colors.primary} />
              <Text fontSize={15} fontWeight="600" color="$color">Sample Words</Text>
            </XStack>

            <YStack gap={6}>
              {samples.slice(0, 20).map((sample, i) => (
                <XStack
                  key={`${sample.word}-${i}`}
                  paddingVertical={8}
                  paddingHorizontal={10}
                  borderRadius={8}
                  backgroundColor={colors.primaryAlpha(0.03)}
                  gap={10}
                  alignItems="center">
                  <Text fontSize={14} fontWeight="600" color="$color" width={80} numberOfLines={1}>
                    {sample.word}
                  </Text>
                  <YStack flex={1} gap={2}>
                    <Text fontSize={11} fontFamily="monospace" color="$color11" numberOfLines={1}>
                      {sample.tokens.join(' · ')}
                    </Text>
                    <Text fontSize={10} color="$color10">
                      [{sample.ids.join(', ')}]
                    </Text>
                  </YStack>
                  <Text fontSize={10} color="$color10">
                    ×{sample.count}
                  </Text>
                </XStack>
              ))}
            </YStack>
          </YStack>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}
