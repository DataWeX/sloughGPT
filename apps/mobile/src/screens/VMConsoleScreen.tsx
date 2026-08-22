import React, {useEffect, useState, useCallback, useRef} from 'react';
import {TextInput, FlatList, Pressable, KeyboardAvoidingView, Platform} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {YStack, XStack, Text} from 'tamagui';
import {useColors} from '../theme/colors';
import {api} from '../services/api-client';
import {StatusBadge} from '../components/StatusBadge';
import {Icon} from '../components/Icon';
import {useServerOutput, type OutputLine} from '../hooks/useServerOutput';

interface VMStatus {
  running: boolean;
  architecture: string | null;
  memory_mb: number;
  disk_image: string | null;
  boot_progress?: number;
}

export function VMConsoleScreen() {
  const colors = useColors();
  const [vmStatus, setVmStatus] = useState<VMStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [input, setInput] = useState('');
  const [commandHistory, setCommandHistory] = useState<string[]>([]);
  const historyIndexRef = useRef(-1);
  const flatListRef = useRef<FlatList>(null);
  const {lines, streaming, paused, clear, togglePause} = useServerOutput({enabled: true, tail: 100});

  const fetchStatus = useCallback(async () => {
    try {
      const data = await api.get<VMStatus>('/vm/status');
      setVmStatus(data);
    } catch {
      setVmStatus(null);
    }
  }, []);

  useEffect(() => {
    fetchStatus().finally(() => setLoading(false));
    const timer = setInterval(fetchStatus, 10000);
    return () => clearInterval(timer);
  }, [fetchStatus]);

  const sendCommand = useCallback(async (cmd: string) => {
    if (!cmd.trim()) return;
    setCommandHistory(prev => [...prev, cmd]);
    historyIndexRef.current = commandHistory.length + 1;
    setInput('');

    try {
      await api.post('/vm/exec', {command: cmd});
    } catch (err) {
      // Errors appear in the output stream
    }
  }, [commandHistory]);

  const handleKeyDown = useCallback(() => {
    if (commandHistory.length === 0) return;
    if (historyIndexRef.current > 0) {
      historyIndexRef.current--;
      setInput(commandHistory[historyIndexRef.current]);
    }
  }, [commandHistory]);

  const handleKeyUp = useCallback(() => {
    if (historyIndexRef.current < commandHistory.length - 1) {
      historyIndexRef.current++;
      setInput(commandHistory[historyIndexRef.current]);
    } else {
      historyIndexRef.current = commandHistory.length;
      setInput('');
    }
  }, [commandHistory]);

  const handleBoot = useCallback(async () => {
    try {
      await api.post('/vm/boot');
    } catch {}
  }, []);

  const handleShutdown = useCallback(async () => {
    try {
      await api.post('/vm/shutdown');
    } catch {}
  }, []);

  const renderLine = useCallback(({item}: {item: OutputLine}) => {
    const severityColor = {
      info: colors.textMuted,
      warning: colors.warning,
      error: colors.error,
      debug: colors.textMuted,
    }[item.severity];

    return (
      <YStack paddingVertical={2} paddingHorizontal={8}>
        <Text fontSize={11} color={severityColor} fontFamily="monospace" selectable>{item.message}</Text>
      </YStack>
    );
  }, [colors]);

  return (
    <SafeAreaView style={{flex: 1, backgroundColor: colors.background}} edges={['top']}>
      <YStack paddingHorizontal={16} paddingVertical={12}>
        <XStack justifyContent="space-between" alignItems="center">
          <YStack>
            <Text fontSize={20} fontWeight="600" color={colors.text}>VM Console</Text>
            <Text fontSize={12} color={colors.textMuted}>x86 Assembly Sandbox + Linux VM</Text>
          </YStack>
          <XStack gap={8}>
            <StatusBadge
              label={vmStatus?.running ? 'Running' : 'Stopped'}
              variant={vmStatus?.running ? 'success' : 'default'}
            />
          </XStack>
        </XStack>
      </YStack>

      {/* VM Status */}
      {vmStatus && (
        <YStack paddingHorizontal={12} paddingBottom={8}>
          <YStack backgroundColor={colors.white} borderRadius={12} borderWidth={0.5} borderColor={colors.border} padding={12} gap={6}>
            <XStack gap={16} flexWrap="wrap">
              <YStack gap={2}>
                <Text fontSize={10} color={colors.textMuted}>ARCH</Text>
                <Text fontSize={12} fontWeight="500" color={colors.text}>{vmStatus.architecture || 'x86'}</Text>
              </YStack>
              <YStack gap={2}>
                <Text fontSize={10} color={colors.textMuted}>MEMORY</Text>
                <Text fontSize={12} fontWeight="500" color={colors.text}>{vmStatus.memory_mb} MB</Text>
              </YStack>
              {vmStatus.boot_progress != null && (
                <YStack gap={2}>
                  <Text fontSize={10} color={colors.textMuted}>BOOT</Text>
                  <Text fontSize={12} fontWeight="500" color={colors.text}>{vmStatus.boot_progress}%</Text>
                </YStack>
              )}
            </XStack>
            <XStack gap={8} marginTop={4}>
              {!vmStatus.running ? (
                <Pressable onPress={handleBoot}>
                  <YStack
                    backgroundColor={colors.successAlpha(0.1)}
                    paddingHorizontal={12}
                    paddingVertical={6}
                    borderRadius={6}
                    pressStyle={{opacity: 0.6}}>
                    <Text fontSize={11} fontWeight="500" color={colors.success}>Boot VM</Text>
                  </YStack>
                </Pressable>
              ) : (
                <Pressable onPress={handleShutdown}>
                  <YStack
                    backgroundColor={colors.errorAlpha(0.1)}
                    paddingHorizontal={12}
                    paddingVertical={6}
                    borderRadius={6}
                    pressStyle={{opacity: 0.6}}>
                    <Text fontSize={11} fontWeight="500" color={colors.error}>Shutdown</Text>
                  </YStack>
                </Pressable>
              )}
              <Pressable onPress={togglePause}>
                <YStack
                  backgroundColor={paused ? colors.warningAlpha(0.1) : colors.muted}
                  paddingHorizontal={12}
                  paddingVertical={6}
                  borderRadius={6}
                  pressStyle={{opacity: 0.6}}>
                  <Text fontSize={11} fontWeight="500" color={paused ? colors.warning : colors.textMuted}>
                    {paused ? 'Resume' : 'Pause'}
                  </Text>
                </YStack>
              </Pressable>
              <Pressable onPress={clear}>
                <YStack
                  backgroundColor={colors.muted}
                  paddingHorizontal={12}
                  paddingVertical={6}
                  borderRadius={6}
                  pressStyle={{opacity: 0.6}}>
                  <Text fontSize={11} fontWeight="500" color={colors.textMuted}>Clear</Text>
                </YStack>
              </Pressable>
            </XStack>
          </YStack>
        </YStack>
      )}

      {/* Console Output */}
      <YStack flex={1} backgroundColor={colors.muted} marginHorizontal={12} borderRadius={12} overflow="hidden">
        <FlatList
          ref={flatListRef}
          data={lines}
          renderItem={renderLine}
          keyExtractor={item => item.id}
          contentContainerStyle={{padding: 4}}
          onContentSizeChange={() => flatListRef.current?.scrollToEnd({animated: false})}
          ListEmptyComponent={
            <YStack padding={16} alignItems="center">
              <Text fontSize={12} color={colors.textMuted}>
                {streaming ? 'Connecting to VM output...' : 'No output yet. Boot the VM to start.'}
              </Text>
            </YStack>
          }
        />

        {/* Input */}
        <XStack
          padding={8}
          borderTopWidth={0.5}
          borderTopColor={colors.border}
          backgroundColor={colors.white}
          alignItems="center"
          gap={8}>
          <Text fontSize={13} fontWeight="700" color={colors.primary} fontFamily="monospace">#</Text>
          <TextInput
            value={input}
            onChangeText={setInput}
            placeholder={vmStatus?.running ? 'Enter command...' : 'VM not running'}
            placeholderTextColor={colors.textMuted}
            editable={vmStatus?.running === true}
            autoCapitalize="none"
            autoCorrect={false}
            selectTextOnFocus
            onSubmitEditing={() => sendCommand(input)}
            returnKeyType="send"
            style={{
              flex: 1,
              fontSize: 12,
              color: colors.text,
              fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
              paddingVertical: 4,
            }}
          />
          <Pressable
            onPress={() => sendCommand(input)}
            disabled={!vmStatus?.running || !input.trim()}>
            <YStack
              width={32}
              height={32}
              borderRadius={8}
              backgroundColor={!vmStatus?.running || !input.trim() ? colors.muted : colors.primary}
              alignItems="center"
              justifyContent="center"
              opacity={!vmStatus?.running || !input.trim() ? 0.4 : 1}>
              <Icon name="send" size={14} color={colors.white} />
            </YStack>
          </Pressable>
        </XStack>
      </YStack>

      {/* Quick Commands */}
      <YStack paddingHorizontal={12} paddingVertical={8} gap={6}>
        <Text fontSize={10} color={colors.textMuted} paddingHorizontal={4}>Quick Commands</Text>
        <XStack gap={6} flexWrap="wrap">
          {['ls', 'cat /proc/cpuinfo', 'free -m', 'df -h', 'uname -a'].map(cmd => (
            <Pressable key={cmd} onPress={() => sendCommand(cmd)}>
              <YStack
                paddingHorizontal={8}
                paddingVertical={4}
                borderRadius={6}
                backgroundColor={colors.muted}
                borderWidth={0.5}
                borderColor={colors.border}
                pressStyle={{opacity: 0.6}}>
                <Text fontSize={10} color={colors.text} fontFamily="monospace">{cmd}</Text>
              </YStack>
            </Pressable>
          ))}
        </XStack>
      </YStack>
    </SafeAreaView>
  );
}
