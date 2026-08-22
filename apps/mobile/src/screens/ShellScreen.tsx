import React, {useEffect, useState, useCallback, useRef} from 'react';
import {TextInput, FlatList, Pressable, KeyboardAvoidingView, Platform} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {YStack, XStack, Text} from 'tamagui';
import {useColors} from '../theme/colors';
import {api} from '../services/api-client';
import {StatusBadge} from '../components/StatusBadge';
import {Icon} from '../components/Icon';

interface ShellEntry {
  id: string;
  type: 'command' | 'output' | 'error';
  content: string;
  timestamp: number;
}

export function ShellScreen() {
  const colors = useColors();
  const [entries, setEntries] = useState<ShellEntry[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [connected, setConnected] = useState(false);
  const flatListRef = useRef<FlatList>(null);
  const inputRef = useRef<TextInput>(null);
  const historyRef = useRef<string[]>([]);
  const historyIndexRef = useRef(-1);

  const addEntry = useCallback((type: ShellEntry['type'], content: string) => {
    setEntries(prev => [...prev, {
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
      type,
      content,
      timestamp: Date.now(),
    }]);
  }, []);

  useEffect(() => {
    api.get('/health').then(() => {
      setConnected(true);
      addEntry('output', 'Connected to SloughGPT server');
    }).catch(() => {
      addEntry('error', 'Could not connect to server');
    });
  }, [addEntry]);

  const executeCommand = useCallback(async (cmd: string) => {
    if (!cmd.trim()) return;

    addEntry('command', cmd);
    historyRef.current.push(cmd);
    historyIndexRef.current = historyRef.current.length;
    setInput('');
    setLoading(true);

    try {
      const result = await api.post<{output: string; exit_code: number}>('/shell/exec', {command: cmd});
      if (result.output) {
        addEntry(result.exit_code === 0 ? 'output' : 'error', result.output);
      }
      if (result.exit_code !== 0) {
        addEntry('error', `Exit code: ${result.exit_code}`);
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Command failed';
      addEntry('error', msg);
    } finally {
      setLoading(false);
    }
  }, [addEntry]);

  const handleKeyDown = useCallback(() => {
    if (historyRef.current.length === 0) return;
    if (historyIndexRef.current > 0) {
      historyIndexRef.current--;
      setInput(historyRef.current[historyIndexRef.current]);
    }
  }, []);

  const handleKeyUp = useCallback(() => {
    if (historyIndexRef.current < historyRef.current.length - 1) {
      historyIndexRef.current++;
      setInput(historyRef.current[historyIndexRef.current]);
    } else {
      historyIndexRef.current = historyRef.current.length;
      setInput('');
    }
  }, []);

  const renderEntry = useCallback(({item}: {item: ShellEntry}) => {
    const isCommand = item.type === 'command';
    const isError = item.type === 'error';

    return (
      <YStack paddingVertical={4} paddingHorizontal={12}>
        {isCommand ? (
          <XStack gap={8} alignItems="flex-start">
            <Text fontSize={13} fontWeight="700" color={colors.primary} fontFamily="monospace">$</Text>
            <Text fontSize={13} fontWeight="500" color={colors.text} fontFamily="monospace" flex={1}>{item.content}</Text>
          </XStack>
        ) : (
          <Text
            fontSize={12}
            color={isError ? colors.error : colors.textMuted}
            fontFamily="monospace"
            selectable>
            {item.content}
          </Text>
        )}
      </YStack>
    );
  }, [colors]);

  return (
    <SafeAreaView style={{flex: 1, backgroundColor: colors.background}} edges={['top']}>
      <YStack paddingHorizontal={16} paddingVertical={12}>
        <XStack justifyContent="space-between" alignItems="center">
          <YStack>
            <Text fontSize={20} fontWeight="600" color={colors.text}>Shell</Text>
            <Text fontSize={12} color={colors.textMuted}>Dait Shell - Interactive Terminal</Text>
          </YStack>
          <StatusBadge
            label={connected ? 'Connected' : 'Disconnected'}
            variant={connected ? 'success' : 'error'}
          />
        </XStack>
      </YStack>

      <YStack flex={1} backgroundColor={colors.muted} marginHorizontal={12} borderRadius={12} overflow="hidden">
        <FlatList
          ref={flatListRef}
          data={entries}
          renderItem={renderEntry}
          keyExtractor={item => item.id}
          contentContainerStyle={{padding: 8}}
          onContentSizeChange={() => flatListRef.current?.scrollToEnd({animated: false})}
          ListEmptyComponent={
            <YStack padding={16} alignItems="center">
              <Text fontSize={12} color={colors.textMuted}>Type a command to get started</Text>
            </YStack>
          }
        />

        <XStack
          padding={8}
          borderTopWidth={0.5}
          borderTopColor={colors.border}
          backgroundColor={colors.white}
          alignItems="center"
          gap={8}>
          <Text fontSize={13} fontWeight="700" color={colors.primary} fontFamily="monospace">$</Text>
          <TextInput
            ref={inputRef}
            value={input}
            onChangeText={setInput}
            placeholder={loading ? 'Running...' : 'Enter command...'}
            placeholderTextColor={colors.textMuted}
            editable={!loading}
            autoCapitalize="none"
            autoCorrect={false}
            selectTextOnFocus
            onSubmitEditing={() => executeCommand(input)}
            returnKeyType="send"
            style={{
              flex: 1,
              fontSize: 13,
              color: colors.text,
              fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
              paddingVertical: 4,
            }}
          />
          <Pressable
            onPress={() => executeCommand(input)}
            disabled={loading || !input.trim()}>
            <YStack
              width={32}
              height={32}
              borderRadius={8}
              backgroundColor={loading || !input.trim() ? colors.muted : colors.primary}
              alignItems="center"
              justifyContent="center"
              opacity={loading || !input.trim() ? 0.4 : 1}>
              <Icon name="send" size={14} color={colors.white} />
            </YStack>
          </Pressable>
        </XStack>
      </YStack>

      {/* Quick Commands */}
      <YStack paddingHorizontal={12} paddingVertical={8} gap={6}>
        <Text fontSize={10} color={colors.textMuted} paddingHorizontal={4}>Quick Commands</Text>
        <XStack gap={6} flexWrap="wrap">
          {['health', 'status', 'models', 'sessions', 'uptime'].map(cmd => (
            <Pressable key={cmd} onPress={() => executeCommand(cmd)}>
              <YStack
                paddingHorizontal={10}
                paddingVertical={5}
                borderRadius={6}
                backgroundColor={colors.muted}
                borderWidth={0.5}
                borderColor={colors.border}
                pressStyle={{opacity: 0.6}}>
                <Text fontSize={11} color={colors.text} fontFamily="monospace">{cmd}</Text>
              </YStack>
            </Pressable>
          ))}
        </XStack>
      </YStack>
    </SafeAreaView>
  );
}
