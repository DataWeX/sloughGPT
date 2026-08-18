import React, {useState, useEffect, useCallback} from 'react';
import {RefreshControl, Pressable, ScrollView} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {YStack, XStack, Text} from 'tamagui';
import {useColors} from '../theme/colors';
import {useModelStore} from '../stores/model-store';
import {api} from '../services/api-client';
import {StatusBadge} from '../components/StatusBadge';
import {Icon} from '../components/Icon';
import {toast} from '../services/toast';
import {triggerHaptic} from '../services/haptics';
import type {SoulInfo, CheckpointInfo} from '../types';

interface SoulDetail {
  name: string;
  description: string;
  traits: string[];
  personality: Record<string, number>;
  born_at?: string;
  training_dataset?: string;
  epochs_trained?: number;
  final_train_loss?: number | null;
  base_model?: string;
}

function traitColor(value: number, colors: ReturnType<typeof import('../theme/colors').useColors>): string {
  if (value >= 0.8) return colors.success;
  if (value >= 0.6) return colors.primary;
  if (value >= 0.4) return colors.warning;
  return colors.textMuted;
}

function traitLabel(key: string): string {
  return key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

export function SoulsScreen() {
  const colors = useColors();

  const {souls, currentSoul, checkpoints, switchSoul, refresh} = useModelStore();
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [selectedSoul, setSelectedSoul] = useState<SoulDetail | null>(null);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [switching, setSwitching] = useState<string | null>(null);

  const loadData = useCallback(async () => {
    await refresh();
  }, [refresh]);

  useEffect(() => {
    loadData().finally(() => setLoading(false));
  }, [loadData]);

  const onRefresh = async () => {
    setRefreshing(true);
    await loadData();
    setRefreshing(false);
  };

  const handleSelectSoul = async (soul: SoulInfo) => {
    if (selectedSoul?.name === soul.name) {
      setSelectedSoul(null);
      return;
    }
    setLoadingDetail(true);
    try {
      const data = await api.get<SoulDetail>(`/souls/${soul.name}`);
      setSelectedSoul(data);
    } catch {
      setSelectedSoul({
        name: soul.name,
        description: soul.description,
        traits: soul.traits || [],
        personality: {},
      });
    } finally {
      setLoadingDetail(false);
    }
  };

  const handleSwitch = async (name: string) => {
    setSwitching(name);
    try {
      await switchSoul(name);
      triggerHaptic('success');
      toast.success(`Switched to ${name}`);
    } catch {
      toast.error('Failed to switch soul');
    } finally {
      setSwitching(null);
    }
  };

  const soulCheckpoints = checkpoints.filter(
    cp => !selectedSoul || cp.soul === selectedSoul.name,
  );

  return (
    <SafeAreaView style={{flex: 1}} edges={['top']}>
      <ScrollView
        style={{flex: 1, backgroundColor: colors.background}}
        contentContainerStyle={{padding: 16, gap: 12}}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}>
        <Text fontSize={24} fontWeight="700" letterSpacing={-0.3} color="$color" paddingBottom={4}>
          Souls
        </Text>

        {souls.length === 0 && !loading ? (
          <YStack
            backgroundColor={colors.white}
            borderRadius={12}
            borderWidth={0.5}
            borderColor={colors.border}
            padding={32}
            alignItems="center"
            gap={8}>
            <Icon name="user" size={28} color={colors.textMuted} />
            <Text fontSize={14} color="$color11" textAlign="center">
              No souls available
            </Text>
            <Text fontSize={12} color="$color10" textAlign="center">
              Souls define the AI's personality and behavior
            </Text>
          </YStack>
        ) : (
          souls.map(soul => {
            const isActive = currentSoul?.name === soul.name;
            const isExpanded = selectedSoul?.name === soul.name;
            const isSwitching = switching === soul.name;

            return (
              <YStack
                key={soul.name}
                backgroundColor={colors.white}
                borderRadius={12}
                borderWidth={0.5}
                borderColor={isActive ? colors.primary + '40' : colors.border}
                overflow="hidden">
                {/* Soul row */}
                <XStack
                  padding={14}
                  gap={12}
                  alignItems="center"
                  onPress={() => handleSelectSoul(soul)}
                  pressStyle={{backgroundColor: colors.primaryAlpha(0.04)}}>
                  <YStack
                    width={40}
                    height={40}
                    borderRadius={12}
                    backgroundColor={isActive ? colors.primary + '18' : colors.primaryAlpha(0.06)}
                    alignItems="center"
                    justifyContent="center">
                    <Icon
                      name="user"
                      size={18}
                      color={isActive ? colors.primary : colors.textMuted}
                    />
                  </YStack>
                  <YStack flex={1}>
                    <XStack alignItems="center" gap={6}>
                      <Text fontSize={15} fontWeight="600" color="$color" numberOfLines={1}>
                        {soul.name}
                      </Text>
                      {isActive && (
                        <StatusBadge label="Active" variant="success" />
                      )}
                    </XStack>
                    {soul.description ? (
                      <Text fontSize={12} color="$color11" numberOfLines={1} marginTop={2}>
                        {soul.description}
                      </Text>
                    ) : null}
                  </YStack>
                  <Icon
                    name={isExpanded ? 'arrow-up' : 'arrow-down'}
                    size={16}
                    color={colors.textMuted}
                  />
                </XStack>

                {/* Expanded detail */}
                {isExpanded && selectedSoul && (
                  <YStack
                    paddingHorizontal={14}
                    paddingBottom={14}
                    gap={12}
                    borderTopWidth={0.5}
                    borderTopColor={colors.border}>
                    {/* Traits */}
                    {selectedSoul.traits.length > 0 && (
                      <YStack gap={6} paddingTop={12}>
                        <Text fontSize={11} fontWeight="600" color="$color10" letterSpacing={0.5}>
                          TRAITS
                        </Text>
                        <XStack gap={6} flexWrap="wrap">
                          {selectedSoul.traits.map(trait => (
                            <YStack
                              key={trait}
                              paddingHorizontal={10}
                              paddingVertical={4}
                              borderRadius={999}
                              backgroundColor={colors.primaryAlpha(0.06)}
                              borderWidth={0.5}
                              borderColor={colors.primaryAlpha(0.12)}>
                              <Text fontSize={11} fontWeight="500" color="$color9">
                                {trait}
                              </Text>
                            </YStack>
                          ))}
                        </XStack>
                      </YStack>
                    )}

                    {/* Personality bars */}
                    {Object.keys(selectedSoul.personality).length > 0 && (
                      <YStack gap={6}>
                        <Text fontSize={11} fontWeight="600" color="$color10" letterSpacing={0.5}>
                          PERSONALITY
                        </Text>
                        {Object.entries(selectedSoul.personality).map(([key, value]) => (
                          <YStack key={key} gap={6}>
                            <XStack justifyContent="space-between" marginBottom={2}>
                              <Text fontSize={12} color="$color11">{traitLabel(key)}</Text>
                              <Text fontSize={12} fontWeight="500" color={traitColor(value, colors)}>
                                {(value * 100).toFixed(0)}%
                              </Text>
                            </XStack>
                            <YStack height={4} borderRadius={2} backgroundColor={colors.primaryAlpha(0.08)}>
                              <YStack
                                height={4}
                                borderRadius={2}
                                backgroundColor={traitColor(value, colors)}
                                style={{width: `${Math.min(value * 100, 100)}%` as any}}
                              />
                            </YStack>
                          </YStack>
                        ))}
                      </YStack>
                    )}

                    {/* Metadata */}
                    {(selectedSoul.born_at || selectedSoul.training_dataset || selectedSoul.epochs_trained) && (
                      <YStack gap={4}>
                        <Text fontSize={11} fontWeight="600" color="$color10" letterSpacing={0.5}>
                          DETAILS
                        </Text>
                        {selectedSoul.born_at && (
                          <XStack justifyContent="space-between">
                            <Text fontSize={12} color="$color11">Created</Text>
                            <Text fontSize={12} color="$color">
                              {new Date(selectedSoul.born_at).toLocaleDateString()}
                            </Text>
                          </XStack>
                        )}
                        {selectedSoul.training_dataset && (
                          <XStack justifyContent="space-between">
                            <Text fontSize={12} color="$color11">Dataset</Text>
                            <Text fontSize={12} color="$color" numberOfLines={1} maxWidth={180}>
                              {selectedSoul.training_dataset}
                            </Text>
                          </XStack>
                        )}
                        {selectedSoul.epochs_trained != null && (
                          <XStack justifyContent="space-between">
                            <Text fontSize={12} color="$color11">Epochs</Text>
                            <Text fontSize={12} color="$color">{selectedSoul.epochs_trained}</Text>
                          </XStack>
                        )}
                        {selectedSoul.final_train_loss != null && (
                          <XStack justifyContent="space-between">
                            <Text fontSize={12} color="$color11">Final Loss</Text>
                            <Text fontSize={12} color="$color">{selectedSoul.final_train_loss.toFixed(4)}</Text>
                          </XStack>
                        )}
                      </YStack>
                    )}

                    {/* Switch button */}
                    {!isActive && (
                      <Pressable
                        onPress={() => handleSwitch(soul.name)}
                        disabled={isSwitching}>
                        {({pressed}) => (
                          <YStack
                            backgroundColor={pressed ? colors.primary + 'CC' : colors.primary}
                            borderRadius={10}
                            paddingVertical={10}
                            alignItems="center"
                            opacity={isSwitching ? 0.6 : 1}>
                            <Text fontSize={14} fontWeight="600" color={colors.white}>
                              {isSwitching ? 'Switching...' : 'Switch to this soul'}
                            </Text>
                          </YStack>
                        )}
                      </Pressable>
                    )}
                  </YStack>
                )}
              </YStack>
            );
          })
        )}

        {/* Checkpoints section */}
        {soulCheckpoints.length > 0 && (
          <YStack gap={8} marginTop={4}>
            <Text fontSize={13} fontWeight="500" color="$color11" paddingHorizontal={4}>
              Checkpoints
            </Text>
            {soulCheckpoints.slice(0, 10).map(cp => (
              <XStack
                key={cp.name}
                backgroundColor={colors.white}
                borderRadius={10}
                borderWidth={0.5}
                borderColor={colors.border}
                padding={12}
                gap={10}
                alignItems="center">
                <YStack flex={1}>
                  <Text fontSize={13} fontWeight="500" color="$color" numberOfLines={1}>
                    {cp.name}
                  </Text>
                  <XStack gap={8} marginTop={2}>
                    {cp.soul && (
                      <Text fontSize={11} color="$color10">{cp.soul}</Text>
                    )}
                    {cp.loss > 0 && (
                      <Text fontSize={11} color="$color10">loss: {cp.loss.toFixed(3)}</Text>
                    )}
                  </XStack>
                </YStack>
                {cp.traits && Object.keys(cp.traits).length > 0 && (
                  <XStack gap={4}>
                    {Object.entries(cp.traits).slice(0, 3).map(([k, v]) => (
                      <YStack
                        key={k}
                        paddingHorizontal={6}
                        paddingVertical={2}
                        borderRadius={4}
                        backgroundColor={traitColor(v as number, colors) + '15'}>
                        <Text fontSize={9} fontWeight="500" color={traitColor(v as number, colors)}>
                          {traitLabel(k)}
                        </Text>
                      </YStack>
                    ))}
                  </XStack>
                )}
              </XStack>
            ))}
          </YStack>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}
