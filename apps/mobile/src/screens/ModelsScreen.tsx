import React, {useEffect, useState, useMemo, useCallback} from 'react';
import {
  ScrollView,
  TextInput,
  RefreshControl,
  ActivityIndicator,
  Modal,
  Pressable,
} from 'react-native';
import {YStack, XStack, Text} from 'tamagui';
import {useColors} from '../theme/colors';
import {SafeAreaView} from 'react-native-safe-area-context';
import {useModelStore} from '../stores/model-store';
import {useHybridStore} from '../stores/hybrid-inference-store';
import {StatusBadge} from '../components/StatusBadge';
import {Icon} from '../components/Icon';
import {triggerHaptic} from '../services/haptics';
import type {ModelInfo} from '../types';
import type {ActiveEngine} from '../types/local-inference';

export function ModelsScreen() {
  const colors = useColors();
  const {
    models,
    currentModel,
    souls,
    currentSoul,
    checkpoints,
    health,
    loading,
    loadingModelId,
    error,
    refresh,
    loadModel,
    unloadModel,
    switchSoul,
    clearError,
  } = useModelStore();
  const [refreshing, setRefreshing] = useState(false);
  const [detailModel, setDetailModel] = useState<ModelInfo | null>(null);
  const [search, setSearch] = useState('');

  const accent = colors.primary;

  const filteredModels = useMemo(() => {
    if (!search.trim()) return models;
    const q = search.toLowerCase();
    return models.filter(m =>
      m.name.toLowerCase().includes(q) ||
      m.type.toLowerCase().includes(q) ||
      (m.description || '').toLowerCase().includes(q) ||
      (m.source || '').toLowerCase().includes(q)
    );
  }, [models, search]);

  useEffect(() => {
    refresh();
  }, []);

  const onRefresh = async () => {
    setRefreshing(true);
    await refresh();
    setRefreshing(false);
  };

  const hybrid = useHybridStore();
  const isLoaded = health?.model_loaded;

  return (
    <SafeAreaView style={{flex: 1}} edges={['top']}>
      <ScrollView
        style={{flex: 1}}
        contentContainerStyle={{padding: 16, gap: 12}}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={accent} />
        }>
        <Text fontSize={26} fontWeight="700" letterSpacing={-0.3} color="$color">
          Models
        </Text>

        {error && (
          <XStack
            backgroundColor={colors.errorAlpha(0.08)}
            padding={12}
            borderRadius={10}
            alignItems="center"
            justifyContent="space-between">
            <Text fontSize={13} color="#EF4444" flex={1}>
              {error}
            </Text>
            <Pressable onPress={() => { triggerHaptic('light'); clearError(); }} accessible={true} accessibilityRole="button" accessibilityLabel="Clear error">
              <Icon name="x" size={14} color="#EF4444" />
            </Pressable>
          </XStack>
        )}

        <YStack
          backgroundColor="$background"
          borderRadius={12}
          borderWidth={0.5}
          borderColor="$borderColor"
          padding={16}
          gap={8}>
          <Text fontSize={16} fontWeight="600" color="$color" marginBottom={12}>
            Active Pipeline
          </Text>
          <XStack alignItems="center" justifyContent="space-between" marginBottom={8}>
            <YStack flex={1}>
              <Text
                fontSize={11}
                color="$color10"
                textTransform="uppercase"
                letterSpacing={0.5}>
                Model
              </Text>
              <Text fontSize={15} color="$color" fontWeight="500">
                {currentModel || 'None loaded'}
              </Text>
            </YStack>
            {isLoaded && <StatusBadge label="Loaded" variant="success" />}
          </XStack>
          <XStack alignItems="center" justifyContent="space-between" marginBottom={8}>
            <YStack flex={1}>
              <Text
                fontSize={11}
                color="$color10"
                textTransform="uppercase"
                letterSpacing={0.5}>
                Personality
              </Text>
              <Text fontSize={15} color="$color" fontWeight="500">
                {currentSoul?.name || 'None'}
              </Text>
            </YStack>
          </XStack>
          {currentSoul && currentSoul.description && (
            <Text fontSize={13} color="$color11" marginTop={4}>
              {currentSoul.description}
            </Text>
          )}
          {currentSoul && currentSoul.traits && currentSoul.traits.length > 0 && (
            <XStack gap={4} marginTop={8} flexWrap="wrap">
              {currentSoul.traits.map(trait => (
                <StatusBadge key={trait} label={trait} variant="info" />
              ))}
            </XStack>
          )}
          {isLoaded && (
            <Pressable onPress={() => { triggerHaptic('medium'); unloadModel(); }} accessible={true} accessibilityRole="button">
              <YStack
                marginTop={12}
                paddingVertical={8}
                paddingHorizontal={12}
                backgroundColor={colors.errorAlpha(0.08)}
                borderRadius={8}
                alignItems="center">
                <Text fontSize={13} color="#EF4444" fontWeight="600">
                  Unload model
                </Text>
              </YStack>
            </Pressable>
          )}
        </YStack>

        <YStack
          backgroundColor="$background"
          borderRadius={12}
          borderWidth={0.5}
          borderColor="$borderColor"
          padding={16}
          gap={8}>
          <Text fontSize={16} fontWeight="600" color="$color" marginBottom={12}>
            On-Device Inference
          </Text>

          <XStack gap={4} marginBottom={12}>
            {(['slonet', 'qwen', 'remote'] as ActiveEngine[]).map(engine => {
              const active = hybrid.activeEngine === engine;
              const label =
                engine === 'slonet' ? 'SloNet' : engine === 'qwen' ? 'Qwen' : 'Server';
              return (
                <Pressable key={engine} style={{flex: 1}} onPress={() => { triggerHaptic('selection'); hybrid.setActiveEngine(engine); }} accessible={true} accessibilityRole="button">
                  <YStack
                    paddingVertical={8}
                    borderRadius={8}
                    backgroundColor={active ? accent : '$background'}
                    borderWidth={0.5}
                    borderColor={active ? accent : '$borderColor'}
                    alignItems="center">
                    <Text
                      fontSize={11}
                      color={active ? 'white' : '$color11'}
                      fontWeight="500">
                      {label}
                    </Text>
                  </YStack>
                </Pressable>
              );
            })}
          </XStack>

          <XStack
            alignItems="center"
            justifyContent="space-between"
            paddingVertical={8}
            borderBottomWidth={0.5}
            borderBottomColor="$borderColor">
            <YStack flex={1} marginRight={8}>
              <Text fontSize={15} color="$color" fontWeight="500">
                SloNet (Baby Transformer)
              </Text>
              <Text fontSize={11} color="$color10" marginTop={2}>
                {hybrid.slonet.loaded
                  ? `Loaded — ${hybrid.slonet.modelName}`
                  : 'Not loaded — fast local completions'}
              </Text>
            </YStack>
            {hybrid.slonet.loaded ? (
              <Pressable onPress={hybrid.unloadSloNet} accessible={true} accessibilityRole="button">
                <YStack paddingHorizontal={12} paddingVertical={6} borderRadius={8} backgroundColor={colors.errorAlpha(0.08)}>
                  <Text fontSize={13} color="#EF4444" fontWeight="600">Unload</Text>
                </YStack>
              </Pressable>
            ) : hybrid.slonet.downloadProgress !== null && hybrid.slonet.downloadProgress < 1 ? (
              <ActivityIndicator size="small" color={accent} />
            ) : (
              <Pressable onPress={() => hybrid.loadSloNet()} accessible={true} accessibilityRole="button">
                <YStack paddingHorizontal={12} paddingVertical={6} borderRadius={8} backgroundColor={accent}>
                  <Text fontSize={13} color="white" fontWeight="600">Load</Text>
                </YStack>
              </Pressable>
            )}
          </XStack>

          <XStack
            alignItems="center"
            justifyContent="space-between"
            paddingVertical={8}
            borderBottomWidth={0.5}
            borderBottomColor="$borderColor">
            <YStack flex={1} marginRight={8}>
              <Text fontSize={15} color="$color" fontWeight="500">
                Qwen 0.5B (GGUF)
              </Text>
              <Text fontSize={11} color="$color10" marginTop={2}>
                {hybrid.qwen.loaded
                  ? 'Loaded — full chat via llama.rn'
                  : hybrid.qwen.downloadProgress !== null && hybrid.qwen.downloadProgress < 1
                  ? `Downloading... ${Math.round(hybrid.qwen.downloadProgress * 100)}%`
                  : 'Not loaded — complex chat, 15-30 tok/s'}
              </Text>
            </YStack>
            {hybrid.qwen.loaded ? (
              <Pressable onPress={async () => hybrid.unloadQwen()} accessible={true} accessibilityRole="button">
                <YStack paddingHorizontal={12} paddingVertical={6} borderRadius={8} backgroundColor={colors.errorAlpha(0.08)}>
                  <Text fontSize={13} color="#EF4444" fontWeight="600">Unload</Text>
                </YStack>
              </Pressable>
            ) : hybrid.qwen.downloadProgress !== null && hybrid.qwen.downloadProgress < 1 ? (
              <YStack width={60} height={6} borderRadius={3} backgroundColor="$borderColor" overflow="hidden">
                <YStack height="100%" backgroundColor={accent} borderRadius={3} width={`${hybrid.qwen.downloadProgress * 100}%`} />
              </YStack>
            ) : (
              <Pressable onPress={() => hybrid.loadQwen()} accessible={true} accessibilityRole="button">
                <YStack paddingHorizontal={12} paddingVertical={6} borderRadius={8} backgroundColor={accent}>
                  <Text fontSize={13} color="white" fontWeight="600">Download</Text>
                </YStack>
              </Pressable>
            )}
          </XStack>

          {hybrid.lastError && (
            <Text fontSize={11} color="#EF4444" marginTop={8}>
              {hybrid.lastError}
            </Text>
          )}
        </YStack>

        {souls.length > 0 && (
          <YStack
            backgroundColor="$background"
            borderRadius={12}
            borderWidth={0.5}
            borderColor="$borderColor"
            padding={16}
            gap={8}>
            <Text fontSize={16} fontWeight="600" color="$color" marginBottom={12}>
              Personalities
            </Text>
            <ScrollView
              horizontal
              showsHorizontalScrollIndicator={false}
              contentContainerStyle={{gap: 8}}>
              {souls.map(soul => {
                const isActive = currentSoul?.name === soul.name;
                return (
                  <Pressable key={soul.name} onPress={() => { triggerHaptic('selection'); switchSoul(soul.name); }} accessible={true} accessibilityRole="button">
                    <YStack
                      paddingHorizontal={12}
                      paddingVertical={8}
                      borderRadius={999}
                      backgroundColor={isActive ? accent : '$background'}
                      borderWidth={0.5}
                      borderColor={isActive ? accent : '$borderColor'}>
                      <Text
                        fontSize={13}
                        color={isActive ? 'white' : '$color11'}
                        fontWeight="500">
                        {soul.name}
                      </Text>
                    </YStack>
                  </Pressable>
                );
              })}
            </ScrollView>
          </YStack>
        )}

        {checkpoints.length > 0 && (
          <YStack
            backgroundColor="$background"
            borderRadius={12}
            borderWidth={0.5}
            borderColor="$borderColor"
            padding={16}
            gap={8}>
            <Text fontSize={16} fontWeight="600" color="$color" marginBottom={12}>
              Trained Versions
            </Text>
            {checkpoints.map(cp => (
              <Pressable key={cp.name} onPress={() => { triggerHaptic('selection'); switchSoul(cp.soul, cp.name); }} accessible={true} accessibilityRole="button">
                <XStack
                  alignItems="center"
                  justifyContent="space-between"
                  paddingVertical={8}
                  borderBottomWidth={0.5}
                  borderBottomColor="$borderColor">
                  <YStack flex={1}>
                    <Text fontSize={15} color="$color" fontWeight="500">
                      {cp.name}
                    </Text>
                    <Text fontSize={11} color="$color10">
                      {cp.loss !== null ? `Loss: ${cp.loss.toFixed(3)}` : ''}{' '}
                      {cp.steps > 0 ? `\u00B7 ${cp.steps} steps` : ''}
                    </Text>
                  </YStack>
                  <StatusBadge label="Use" variant="info" />
                </XStack>
              </Pressable>
            ))}
          </YStack>
        )}

        <YStack
          backgroundColor="$background"
          borderRadius={12}
          borderWidth={0.5}
          borderColor="$borderColor"
          padding={16}
          gap={8}>
          <Text fontSize={16} fontWeight="600" color="$color" marginBottom={12}>
            Available Models
          </Text>
          {models.length > 3 && (
            <TextInput
              value={search}
              onChangeText={setSearch}
              placeholder="Search models..."
              placeholderTextColor={colors.textMuted}
              returnKeyType="search"
              style={{
                fontSize: 14,
                color: colors.text,
                backgroundColor: colors.primaryAlpha(0.04),
                borderRadius: 10,
                paddingHorizontal: 14,
                paddingVertical: 10,
                marginBottom: 12,
                borderWidth: 0.5,
                borderColor: colors.primaryAlpha(0.12),
              }}
            />
          )}
          {filteredModels.length === 0 && !loading && (
            <Text fontSize={13} color="$color10" textAlign="center" paddingVertical={16}>
              {search ? 'No models match your search' : 'No models found'}
            </Text>
          )}
          {filteredModels.map(model => {
            const isLoading = loadingModelId === model.id;
            return (
              <Pressable key={model.id} onPress={() => setDetailModel(model)} accessible={true} accessibilityRole="button">
                <XStack
                  alignItems="center"
                  justifyContent="space-between"
                  paddingVertical={8}
                  borderBottomWidth={0.5}
                  borderBottomColor="$borderColor">
                  <YStack flex={1}>
                    <Text fontSize={15} color="$color" fontWeight="500">
                      {model.name}
                    </Text>
                    <Text fontSize={11} color="$color10">
                      {model.size_gb
                        ? `${model.size_gb.toFixed(1)} GB`
                        : model.size_mb
                        ? `${model.size_mb} MB`
                        : model.params || model.type}
                    </Text>
                  </YStack>
                  {isLoading ? (
                    <ActivityIndicator size="small" color={accent} />
                  ) : model.loaded ? (
                    <StatusBadge label="Loaded" variant="success" />
                  ) : (
                    <Pressable onPress={(e) => { e.stopPropagation(); triggerHaptic('medium'); loadModel(model.id); }}>
                      <YStack paddingHorizontal={12} paddingVertical={6} borderRadius={8} backgroundColor={accent}>
                        <Text fontSize={13} color="white" fontWeight="600">Load</Text>
                      </YStack>
                    </Pressable>
                  )}
                </XStack>
              </Pressable>
            );
          })}
        </YStack>
      </ScrollView>

      {/* Model detail bottom sheet */}
      <Modal visible={!!detailModel} animationType="slide" transparent>
        <YStack flex={1} backgroundColor={colors.overlay(0.4)} justifyContent="flex-end">
          <YStack
            backgroundColor="$background"
            borderTopLeftRadius={24}
            borderTopRightRadius={24}
            maxHeight="80%">
            <XStack
              alignItems="center"
              justifyContent="space-between"
              paddingHorizontal={20}
              paddingVertical={16}
              borderBottomWidth={0.5}
              borderBottomColor="$borderColor">
              <Text fontSize={20} fontWeight="600" color="$color" flex={1}>
                {detailModel?.name}
              </Text>
              <Pressable onPress={() => setDetailModel(null)} accessible={true} accessibilityRole="button" accessibilityLabel="Close modal">
                <YStack width={28} height={28} borderRadius={9} alignItems="center" justifyContent="center">
                  <Icon name="x" size={16} color={colors.textSecondary} />
                </YStack>
              </Pressable>
            </XStack>
            <YStack paddingHorizontal={20} paddingVertical={16} gap={12}>
              {detailModel?.description && (
                <Text fontSize={15} color="$color11">
                  {detailModel.description}
                </Text>
              )}
              <XStack alignItems="center" justifyContent="space-between">
                <Text fontSize={13} color="$color10">Type</Text>
                <Text fontSize={15} color="$color" fontWeight="500">
                  {detailModel?.type || '\u2014'}
                </Text>
              </XStack>
              <XStack alignItems="center" justifyContent="space-between">
                <Text fontSize={13} color="$color10">Parameters</Text>
                <Text fontSize={15} color="$color" fontWeight="500">
                  {detailModel?.params || '\u2014'}
                </Text>
              </XStack>
              <XStack alignItems="center" justifyContent="space-between">
                <Text fontSize={13} color="$color10">Size</Text>
                <Text fontSize={15} color="$color" fontWeight="500">
                  {detailModel?.size_gb
                    ? `${detailModel.size_gb.toFixed(1)} GB`
                    : detailModel?.size_mb
                    ? `${detailModel.size_mb} MB`
                    : '\u2014'}
                </Text>
              </XStack>
              <XStack alignItems="center" justifyContent="space-between">
                <Text fontSize={13} color="$color10">Source</Text>
                <Text fontSize={15} color="$color" fontWeight="500">
                  {detailModel?.source || '\u2014'}
                </Text>
              </XStack>
              <XStack alignItems="center" justifyContent="space-between">
                <Text fontSize={13} color="$color10">Status</Text>
                <StatusBadge
                  label={detailModel?.loaded ? 'Loaded' : 'Available'}
                  variant={detailModel?.loaded ? 'success' : 'default'}
                />
              </XStack>
              {detailModel?.tags && detailModel.tags.length > 0 && (
                <XStack gap={4} flexWrap="wrap">
                  {detailModel.tags.map(tag => (
                    <StatusBadge key={tag} label={tag} variant="info" />
                  ))}
                </XStack>
              )}
            </YStack>
            <YStack
              paddingHorizontal={20}
              paddingVertical={16}
              borderTopWidth={0.5}
              borderTopColor="$borderColor">
              {detailModel?.loaded ? (
                <Pressable onPress={() => { triggerHaptic('medium'); unloadModel(); setDetailModel(null); }} accessible={true} accessibilityRole="button">
                  <YStack backgroundColor={colors.errorAlpha(0.08)} paddingVertical={12} borderRadius={10} alignItems="center">
                    <Text fontSize={15} color="#EF4444" fontWeight="600">Unload Model</Text>
                  </YStack>
                </Pressable>
              ) : (
                <Pressable onPress={() => { triggerHaptic('medium'); if (detailModel) loadModel(detailModel.id); setDetailModel(null); }} accessible={true} accessibilityRole="button">
                  <YStack backgroundColor={accent} paddingVertical={12} borderRadius={10} alignItems="center">
                    <Text fontSize={15} color="white" fontWeight="600">Load Model</Text>
                  </YStack>
                </Pressable>
              )}
            </YStack>
          </YStack>
        </YStack>
      </Modal>
    </SafeAreaView>
  );
}
