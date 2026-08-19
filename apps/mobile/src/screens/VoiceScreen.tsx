import React, {useEffect, useState, useCallback, useRef} from 'react';
import {FlatList, Pressable, RefreshControl, TextInput as RNTextInput} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {YStack, XStack, Text} from 'tamagui';
import {Audio} from 'expo-av';
import {useColors} from '../theme/colors';
import {api} from '../services/api-client';
import {Icon} from '../components/Icon';
import {StatusBadge} from '../components/StatusBadge';
import {triggerHaptic} from '../services/haptics';
import {toast} from '../services/toast';

interface VoiceStatus {
  tts_available: boolean;
  model_name: string | null;
  tts_calls: number;
  fallback: string;
  error: string | null;
}

interface TTSResponse {
  audio: string;
  sample_rate: number;
  duration_ms: number;
  backend: string;
}

export function VoiceScreen() {
  const colors = useColors();
  const [status, setStatus] = useState<VoiceStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [inputText, setInputText] = useState('');
  const [generating, setGenerating] = useState(false);
  const [playing, setPlaying] = useState(false);
  const soundRef = useRef<Audio.Sound | null>(null);

  const fetchStatus = useCallback(async () => {
    try {
      const data = await api.get<VoiceStatus>('/voice/status');
      setStatus(data);
    } catch {
      setStatus(null);
    }
  }, []);

  useEffect(() => {
    fetchStatus().finally(() => setLoading(false));
    return () => {
      soundRef.current?.unloadAsync().catch(() => {});
    };
  }, [fetchStatus]);

  const onRefresh = async () => {
    setRefreshing(true);
    await fetchStatus();
    setRefreshing(false);
  };

  const handleGenerate = async () => {
    if (!inputText.trim()) return;
    try {
      setGenerating(true);
      triggerHaptic('light');

      if (soundRef.current) {
        await soundRef.current.unloadAsync().catch(() => {});
        soundRef.current = null;
      }

      const result = await api.post<TTSResponse>('/voice/tts', {text: inputText.trim()});

      if (!result.audio) {
        toast.error('No audio returned');
        return;
      }

      const {sound} = await Audio.Sound.createAsync(
        {uri: `data:audio/wav;base64,${result.audio}`},
        {shouldPlay: true},
      );
      soundRef.current = sound;
      setPlaying(true);

      sound.setOnPlaybackStatusUpdate((status) => {
        if (status.isLoaded && status.didJustFinish) {
          setPlaying(false);
          sound.unloadAsync().catch(() => {});
          soundRef.current = null;
        }
      });

      triggerHaptic('success');
      toast.success(`Played ${((result.duration_ms || 0) / 1000).toFixed(1)}s audio`);
      setInputText('');
      await fetchStatus();
    } catch {
      toast.error('TTS generation failed');
    } finally {
      setGenerating(false);
    }
  };

  return (
    <SafeAreaView style={{flex: 1, backgroundColor: colors.background}} edges={['top']}>
      <XStack paddingHorizontal={16} paddingVertical={12} alignItems="center" justifyContent="space-between">
        <Text fontSize={20} fontWeight="600" color={colors.text}>Voice</Text>
        <Pressable onPress={onRefresh}>
          <Icon name="refresh-cw" size={18} color={colors.primary} />
        </Pressable>
      </XStack>

      {loading ? (
        <YStack flex={1} alignItems="center" justifyContent="center">
          <StatusBadge label="Loading..." variant="info" />
        </YStack>
      ) : (
        <FlatList
          data={[]}
          renderItem={() => null}
          ListHeaderComponent={
            <YStack padding={16} gap={12}>
              {/* TTS Status */}
              <YStack padding={14} borderRadius={10} backgroundColor={colors.white} borderWidth={0.5} borderColor={colors.border} gap={10}>
                <XStack justifyContent="space-between" alignItems="center">
                  <Text fontSize={15} fontWeight="600" color={colors.text}>TTS Status</Text>
                  <StatusBadge label={status?.tts_available ? 'Available' : 'Offline'} variant={status?.tts_available ? 'success' : 'error'} />
                </XStack>
                <XStack gap={16}>
                  <YStack gap={2}>
                    <Text fontSize={11} color={colors.textMuted}>Model</Text>
                    <Text fontSize={13} fontWeight="500" color={colors.text}>{status?.model_name || '—'}</Text>
                  </YStack>
                  <YStack gap={2}>
                    <Text fontSize={11} color={colors.textMuted}>TTS Calls</Text>
                    <Text fontSize={13} fontWeight="500" color={colors.text}>{status?.tts_calls ?? 0}</Text>
                  </YStack>
                  <YStack gap={2}>
                    <Text fontSize={11} color={colors.textMuted}>Fallback</Text>
                    <Text fontSize={13} fontWeight="500" color={colors.text}>{status?.fallback || '—'}</Text>
                  </YStack>
                </XStack>
                {status?.error && (
                  <XStack padding={8} borderRadius={6} backgroundColor={colors.errorAlpha(0.1)} gap={6} alignItems="center">
                    <Icon name="triangle-alert" size={14} color={colors.error} />
                    <Text fontSize={12} color={colors.error} flex={1}>{status.error}</Text>
                  </XStack>
                )}
              </YStack>

              {/* Test TTS */}
              <YStack padding={14} borderRadius={10} backgroundColor={colors.white} borderWidth={0.5} borderColor={colors.border} gap={8}>
                <Text fontSize={15} fontWeight="600" color={colors.text}>Test TTS</Text>
                <RNTextInput
                  value={inputText}
                  onChangeText={setInputText}
                  placeholder="Enter text to synthesize..."
                  placeholderTextColor={colors.textMuted}
                  multiline
                  numberOfLines={3}
                  style={{
                    borderWidth: 1,
                    borderColor: colors.border,
                    borderRadius: 8,
                    padding: 10,
                    fontSize: 14,
                    color: colors.text,
                    backgroundColor: colors.background,
                    minHeight: 72,
                    textAlignVertical: 'top',
                  }}
                />
                <Pressable onPress={handleGenerate} disabled={!inputText.trim() || generating || playing}>
                  <XStack padding={10} borderRadius={8} backgroundColor={inputText.trim() && !generating && !playing ? colors.primary : colors.border} alignItems="center" justifyContent="center" gap={6}>
                    <Icon name={generating ? 'refresh-cw' : playing ? 'volume-2' : 'music'} size={16} color="white" />
                    <Text fontSize={13} fontWeight="600" color="white">{generating ? 'Generating...' : playing ? 'Playing...' : 'Generate & Play'}</Text>
                  </XStack>
                </Pressable>
              </YStack>

              {/* About */}
              <YStack padding={14} borderRadius={10} backgroundColor={colors.white} borderWidth={0.5} borderColor={colors.border} gap={6}>
                <Text fontSize={15} fontWeight="600" color={colors.text}>About</Text>
                <Text fontSize={13} color={colors.textMuted} lineHeight={18}>
                  Voice synthesis converts text to speech using server-side TTS models. Generated audio can be played back or shared.
                </Text>
              </YStack>
            </YStack>
          }
          contentContainerStyle={{paddingBottom: 32}}
          refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}
        />
      )}
    </SafeAreaView>
  );
}
