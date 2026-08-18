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

interface CompanionData {
  active_preset: string | null;
  warmth: number;
  curiosity: number;
  playfulness: number;
  confidence: number;
  empathy: number;
  system_prompt: string;
  presets: string[];
}

const TRAIT_DEFAULTS = {warmth: 0.5, curiosity: 0.5, playfulness: 0.5, confidence: 0.5, empathy: 0.5};

function TraitSlider({label, value, onChange}: {label: string; value: number; onChange: (v: number) => void}) {
  const colors = useColors();
  const steps = [0, 0.25, 0.5, 0.75, 1.0];
  return (
    <YStack gap={4}>
      <XStack justifyContent="space-between" alignItems="center">
        <Text fontSize={13} color={colors.textMuted}>{label}</Text>
        <Text fontSize={13} fontWeight="500" color={colors.text}>{value.toFixed(2)}</Text>
      </XStack>
      <XStack gap={4}>
        {steps.map(s => (
          <Pressable key={s} onPress={() => onChange(s)} style={{flex: 1}}>
            <YStack height={8} borderRadius={4} backgroundColor={s <= value ? colors.primary : colors.border} />
          </Pressable>
        ))}
      </XStack>
    </YStack>
  );
}

export function CompanionScreen() {
  const colors = useColors();
  const [data, setData] = useState<CompanionData | null>(null);
  const [traits, setTraits] = useState(TRAIT_DEFAULTS);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [saving, setSaving] = useState(false);
  const [chatInput, setChatInput] = useState('');
  const [chatResponse, setChatResponse] = useState('');
  const [chatting, setChatging] = useState(false);

  const fetchData = useCallback(async () => {
    try {
      const [companion, prompt] = await Promise.all([
        api.get<CompanionData>('/companion/').catch(() => null),
        api.get<{prompt: string}>('/companion/prompt').catch(() => ({prompt: ''})),
      ]);
      if (companion) {
        setData(companion);
        setTraits({
          warmth: companion.warmth,
          curiosity: companion.curiosity,
          playfulness: companion.playfulness,
          confidence: companion.confidence,
          empathy: companion.empathy,
        });
      }
    } catch {
      // handled above
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

  const handleSave = async () => {
    try {
      setSaving(true);
      triggerHaptic('light');
      await api.post('/companion/personality', traits);
      triggerHaptic('success');
      toast.success('Personality saved');
      await fetchData();
    } catch {
      toast.error('Save failed');
    } finally {
      setSaving(false);
    }
  };

  const handlePreset = async (preset: string) => {
    try {
      triggerHaptic('light');
      await api.post('/companion/preset', {preset});
      triggerHaptic('success');
      toast.success(`Applied "${preset}" preset`);
      await fetchData();
    } catch {
      toast.error('Preset failed');
    }
  };

  const handleChat = async () => {
    if (!chatInput.trim()) return;
    try {
      setChatging(true);
      triggerHaptic('light');
      const result = await api.post<{response: string}>('/companion/chat', {message: chatInput.trim()});
      setChatResponse(result.response);
      setChatInput('');
    } catch {
      toast.error('Chat failed');
    } finally {
      setChatging(false);
    }
  };

  const avgTrait = data ? ((data.warmth + data.curiosity + data.playfulness + data.confidence + data.empathy) / 5).toFixed(2) : '—';

  return (
    <SafeAreaView style={{flex: 1, backgroundColor: colors.background}} edges={['top']}>
      <XStack paddingHorizontal={16} paddingVertical={12} alignItems="center" justifyContent="space-between">
        <Text fontSize={20} fontWeight="600" color={colors.text}>Companion</Text>
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
              {/* KPI */}
              <XStack gap={8}>
                <YStack flex={1} padding={12} borderRadius={10} backgroundColor={colors.white} borderWidth={0.5} borderColor={colors.border} gap={4} alignItems="center">
                  <Text fontSize={11} color={colors.textMuted}>Preset</Text>
                  <Text fontSize={14} fontWeight="600" color={colors.text}>{data?.active_preset || 'Custom'}</Text>
                </YStack>
                <YStack flex={1} padding={12} borderRadius={10} backgroundColor={colors.white} borderWidth={0.5} borderColor={colors.border} gap={4} alignItems="center">
                  <Text fontSize={11} color={colors.textMuted}>Avg Trait</Text>
                  <Text fontSize={14} fontWeight="600" color={colors.primary}>{avgTrait}</Text>
                </YStack>
              </XStack>

              {/* Presets */}
              {data?.presets && data.presets.length > 0 && (
                <YStack padding={14} borderRadius={10} backgroundColor={colors.white} borderWidth={0.5} borderColor={colors.border} gap={8}>
                  <Text fontSize={15} fontWeight="600" color={colors.text}>Presets</Text>
                  <XStack gap={6} flexWrap="wrap">
                    {data.presets.map(p => (
                      <Pressable key={p} onPress={() => handlePreset(p)}>
                        <XStack paddingHorizontal={10} paddingVertical={5} borderRadius={6} backgroundColor={data.active_preset === p ? colors.primary : colors.primaryAlpha(0.1)} gap={4} alignItems="center">
                          <Text fontSize={12} fontWeight="500" color={data.active_preset === p ? 'white' : colors.primary}>{p}</Text>
                        </XStack>
                      </Pressable>
                    ))}
                  </XStack>
                </YStack>
              )}

              {/* Personality Traits */}
              <YStack padding={14} borderRadius={10} backgroundColor={colors.white} borderWidth={0.5} borderColor={colors.border} gap={10}>
                <Text fontSize={15} fontWeight="600" color={colors.text}>Personality Traits</Text>
                {(Object.keys(traits) as Array<keyof typeof traits>).map(key => (
                  <TraitSlider
                    key={key}
                    label={key.charAt(0).toUpperCase() + key.slice(1)}
                    value={traits[key]}
                    onChange={v => setTraits(prev => ({...prev, [key]: v}))}
                  />
                ))}
                <Pressable onPress={handleSave} disabled={saving}>
                  <XStack padding={10} borderRadius={8} backgroundColor={!saving ? colors.primary : colors.border} alignItems="center" justifyContent="center" gap={6}>
                    <Icon name="save" size={16} color="white" />
                    <Text fontSize={13} fontWeight="600" color="white">{saving ? 'Saving...' : 'Save Personality'}</Text>
                  </XStack>
                </Pressable>
              </YStack>

              {/* System Prompt */}
              {data?.system_prompt && (
                <YStack padding={14} borderRadius={10} backgroundColor={colors.white} borderWidth={0.5} borderColor={colors.border} gap={6}>
                  <Text fontSize={15} fontWeight="600" color={colors.text}>System Prompt</Text>
                  <YStack padding={10} borderRadius={6} backgroundColor={colors.background}>
                    <Text fontSize={12} fontFamily="monospace" color={colors.textMuted} lineHeight={18}>{data.system_prompt}</Text>
                  </YStack>
                </YStack>
              )}

              {/* Test Chat */}
              <YStack padding={14} borderRadius={10} backgroundColor={colors.white} borderWidth={0.5} borderColor={colors.border} gap={8}>
                <Text fontSize={15} fontWeight="600" color={colors.text}>Test Chat</Text>
                <RNTextInput
                  value={chatInput}
                  onChangeText={setChatInput}
                  placeholder="Say something to the companion..."
                  placeholderTextColor={colors.textMuted}
                  style={{
                    borderWidth: 1,
                    borderColor: colors.border,
                    borderRadius: 8,
                    padding: 10,
                    fontSize: 14,
                    color: colors.text,
                    backgroundColor: colors.background,
                  }}
                />
                <Pressable onPress={handleChat} disabled={!chatInput.trim() || chatting}>
                  <XStack padding={10} borderRadius={8} backgroundColor={chatInput.trim() && !chatting ? colors.primary : colors.border} alignItems="center" justifyContent="center" gap={6}>
                    <Icon name="message-circle" size={16} color="white" />
                    <Text fontSize={13} fontWeight="600" color="white">{chatting ? 'Thinking...' : 'Send'}</Text>
                  </XStack>
                </Pressable>
                {chatResponse ? (
                  <YStack padding={10} borderRadius={6} backgroundColor={colors.primaryAlpha(0.05)} gap={4}>
                    <Text fontSize={13} color={colors.text} lineHeight={18}>{chatResponse}</Text>
                  </YStack>
                ) : null}
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
