import React, {useEffect, useState, useMemo, useCallback} from 'react';
import {
  ScrollView,
  TextInput,
  RefreshControl,
  ActivityIndicator,
  Modal,
} from 'react-native';
import {YStack, XStack, Text} from 'tamagui';
import {SafeAreaView} from 'react-native-safe-area-context';
import {useModelStore} from '../stores/model-store';
import {useHybridStore} from '../stores/hybrid-inference-store';
import {StatusBadge} from '../components/StatusBadge';
import type {ModelInfo} from '../types';
import type {ActiveEngine} from '../types/local-inference';

export function ModelsScreen() {
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
    <SafeAreaView style={{flex: 1, backgroundColor: '$background'}} edges={['top']}>
      <ScrollView
        style={{flex: 1}}
        contentContainerStyle={{padding: 16, gap: 12}}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }>
        <Text fontSize={26} fontWeight="700" letterSpacing={-0.3} color="$color">
          Models
        </Text>

        {error && (
          <XStack
            backgroundColor="#FDE8E8"
            padding={12}
            borderRadius={8}
            alignItems="center"
            justifyContent="space-between">
            <Text fontSize={13} color="#D44C56" flex={1}>
              {error}
            </Text>
            <YStack onPress={clearError} pressStyle={{opacity: 0.7}}>
              <Text fontSize={15} color="#D44C56" fontWeight="600" marginLeft={8}>
                ×
              </Text>
            </YStack>
          </XStack>
        )}

        <YStack
          backgroundColor="$background"
          borderRadius={12}
          borderWidth={1}
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
            <YStack
              marginTop={12}
              paddingVertical={8}
              paddingHorizontal={12}
              backgroundColor="#D44C5615"
              borderRadius={8}
              alignItems="center"
              onPress={unloadModel}
              pressStyle={{opacity: 0.7}}>
              <Text fontSize={13} color="#D44C56" fontWeight="600">
                Unload model
              </Text>
            </YStack>
          )}
        </YStack>

        <YStack
          backgroundColor="$background"
          borderRadius={12}
          borderWidth={1}
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
                engine === 'slonet'
                  ? 'SloNet'
                  : engine === 'qwen'
                  ? 'Qwen'
                  : 'Server';
              return (
                <YStack
                  key={engine}
                  flex={1}
                  paddingVertical={8}
                  borderRadius={8}
                  backgroundColor={active ? '$color9' : '$background'}
                  borderWidth={1}
                  borderColor={active ? '$color9' : '$borderColor'}
                  alignItems="center"
                  onPress={() => hybrid.setActiveEngine(engine)}
                  pressStyle={{opacity: 0.7}}>
                  <Text
                    fontSize={11}
                    color={active ? 'white' : '$color11'}
                    fontWeight="500">
                    {label}
                  </Text>
                </YStack>
              );
            })}
          </XStack>

          <XStack
            alignItems="center"
            justifyContent="space-between"
            paddingVertical={8}
            borderBottomWidth={1}
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
              <YStack
                paddingHorizontal={12}
                paddingVertical={6}
                borderRadius={8}
                backgroundColor="#D44C5615"
                onPress={hybrid.unloadSloNet}
                pressStyle={{opacity: 0.7}}>
                <Text fontSize={13} color="#D44C56" fontWeight="600">
                  Unload
                </Text>
              </YStack>
            ) : hybrid.slonet.downloadProgress !== null &&
              hybrid.slonet.downloadProgress < 1 ? (
              <ActivityIndicator size="small" color="#7C52C4" />
            ) : (
              <YStack
                paddingHorizontal={12}
                paddingVertical={6}
                borderRadius={8}
                backgroundColor="$color9"
                onPress={() => hybrid.loadSloNet()}
                pressStyle={{opacity: 0.7}}>
                <Text fontSize={13} color="white" fontWeight="600">
                  Load
                </Text>
              </YStack>
            )}
          </XStack>

          <XStack
            alignItems="center"
            justifyContent="space-between"
            paddingVertical={8}
            borderBottomWidth={1}
            borderBottomColor="$borderColor">
            <YStack flex={1} marginRight={8}>
              <Text fontSize={15} color="$color" fontWeight="500">
                Qwen 0.5B (GGUF)
              </Text>
              <Text fontSize={11} color="$color10" marginTop={2}>
                {hybrid.qwen.loaded
                  ? 'Loaded — full chat via llama.rn'
                  : hybrid.qwen.downloadProgress !== null &&
                    hybrid.qwen.downloadProgress < 1
                  ? `Downloading... ${Math.round(
                      hybrid.qwen.downloadProgress * 100,
                    )}%`
                  : 'Not loaded — complex chat, 15-30 tok/s'}
              </Text>
            </YStack>
            {hybrid.qwen.loaded ? (
              <YStack
                paddingHorizontal={12}
                paddingVertical={6}
                borderRadius={8}
                backgroundColor="#D44C5615"
                onPress={async () => hybrid.unloadQwen()}
                pressStyle={{opacity: 0.7}}>
                <Text fontSize={13} color="#D44C56" fontWeight="600">
                  Unload
                </Text>
              </YStack>
            ) : hybrid.qwen.downloadProgress !== null &&
              hybrid.qwen.downloadProgress < 1 ? (
              <YStack
                width={60}
                height={6}
                borderRadius={3}
                backgroundColor="$borderColor"
                overflow="hidden">
                <YStack
                  height="100%"
                  backgroundColor="$color9"
                  borderRadius={3}
                  width={`${hybrid.qwen.downloadProgress * 100}%`}
                />
              </YStack>
            ) : (
              <YStack
                paddingHorizontal={12}
                paddingVertical={6}
                borderRadius={8}
                backgroundColor="$color9"
                onPress={() => hybrid.loadQwen()}
                pressStyle={{opacity: 0.7}}>
                <Text fontSize={13} color="white" fontWeight="600">
                  Download
                </Text>
              </YStack>
            )}
          </XStack>

          {hybrid.lastError && (
            <Text fontSize={11} color="#D44C56" marginTop={8}>
              {hybrid.lastError}
            </Text>
          )}
        </YStack>

        {souls.length > 0 && (
          <YStack
            backgroundColor="$background"
            borderRadius={12}
            borderWidth={1}
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
                  <YStack
                    key={soul.name}
                    paddingHorizontal={12}
                    paddingVertical={8}
                    borderRadius={9999}
                    backgroundColor={isActive ? '$color9' : '$background'}
                    borderWidth={1}
                    borderColor={isActive ? '$color9' : '$borderColor'}
                    onPress={() => switchSoul(soul.name)}
                    pressStyle={{opacity: 0.7}}>
                    <Text
                      fontSize={13}
                      color={isActive ? 'white' : '$color11'}
                      fontWeight="500">
                      {soul.name}
                    </Text>
                  </YStack>
                );
              })}
            </ScrollView>
          </YStack>
        )}

        {checkpoints.length > 0 && (
          <YStack
            backgroundColor="$background"
            borderRadius={12}
            borderWidth={1}
            borderColor="$borderColor"
            padding={16}
            gap={8}>
            <Text fontSize={16} fontWeight="600" color="$color" marginBottom={12}>
              Trained Versions
            </Text>
            {checkpoints.map(cp => (
              <XStack
                key={cp.name}
                alignItems="center"
                justifyContent="space-between"
                paddingVertical={8}
                borderBottomWidth={1}
                borderBottomColor="$borderColor"
                onPress={() => switchSoul(cp.soul, cp.name)}
                pressStyle={{opacity: 0.7}}>
                <YStack flex={1}>
                  <Text fontSize={15} color="$color" fontWeight="500">
                    {cp.name}
                  </Text>
                  <Text fontSize={11} color="$color10">
                    {cp.loss !== null ? `Loss: ${cp.loss.toFixed(3)}` : ''}{' '}
                    {cp.steps > 0 ? `· ${cp.steps} steps` : ''}
                  </Text>
                </YStack>
                <StatusBadge label="Use" variant="info" />
              </XStack>
            ))}
          </YStack>
        )}

        <YStack
          backgroundColor="$background"
          borderRadius={12}
          borderWidth={1}
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
              placeholderTextColor="#9B95A8"
              returnKeyType="search"
              style={{
                fontSize: 15,
                color: '#1A1625',
                backgroundColor: 'rgba(124, 82, 196, 0.04)',
                borderRadius: 8,
                paddingHorizontal: 12,
                paddingVertical: 8,
                marginBottom: 12,
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
              <XStack
                key={model.id}
                alignItems="center"
                justifyContent="space-between"
                paddingVertical={8}
                borderBottomWidth={1}
                borderBottomColor="$borderColor"
                onPress={() => setDetailModel(model)}
                pressStyle={{opacity: 0.7}}>
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
                  <ActivityIndicator size="small" color="#7C52C4" />
                ) : model.loaded ? (
                  <StatusBadge label="Loaded" variant="success" />
                ) : (
                  <YStack
                    paddingHorizontal={12}
                    paddingVertical={6}
                    borderRadius={8}
                    backgroundColor="$color9"
                    onPress={(e: any) => {
                      e.stopPropagation();
                      loadModel(model.id);
                    }}
                    pressStyle={{opacity: 0.7}}>
                    <Text fontSize={13} color="white" fontWeight="600">
                      Load
                    </Text>
                  </YStack>
                )}
              </XStack>
            );
          })}
        </YStack>
      </ScrollView>

      <Modal visible={!!detailModel} animationType="slide" transparent>
        <YStack flex={1} backgroundColor="rgba(0,0,0,0.4)" justifyContent="flex-end">
          <YStack
            backgroundColor="$background"
            borderTopLeftRadius={16}
            borderTopRightRadius={16}
            maxHeight="80%">
            <XStack
              alignItems="center"
              justifyContent="space-between"
              paddingHorizontal={20}
              paddingVertical={16}
              borderBottomWidth={1}
              borderBottomColor="$borderColor">
              <Text fontSize={20} fontWeight="600" color="$color" flex={1}>
                {detailModel?.name}
              </Text>
              <YStack onPress={() => setDetailModel(null)} pressStyle={{opacity: 0.7}}>
                <Text fontSize={24} color="$color10" padding={4}>
                  ×
                </Text>
              </YStack>
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
                  {detailModel?.type || '—'}
                </Text>
              </XStack>
              <XStack alignItems="center" justifyContent="space-between">
                <Text fontSize={13} color="$color10">Parameters</Text>
                <Text fontSize={15} color="$color" fontWeight="500">
                  {detailModel?.params || '—'}
                </Text>
              </XStack>
              <XStack alignItems="center" justifyContent="space-between">
                <Text fontSize={13} color="$color10">Size</Text>
                <Text fontSize={15} color="$color" fontWeight="500">
                  {detailModel?.size_gb
                    ? `${detailModel.size_gb.toFixed(1)} GB`
                    : detailModel?.size_mb
                    ? `${detailModel.size_mb} MB`
                    : '—'}
                </Text>
              </XStack>
              <XStack alignItems="center" justifyContent="space-between">
                <Text fontSize={13} color="$color10">Source</Text>
                <Text fontSize={15} color="$color" fontWeight="500">
                  {detailModel?.source || '—'}
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
              borderTopWidth={1}
              borderTopColor="$borderColor">
              {detailModel?.loaded ? (
                <YStack
                  backgroundColor="#D44C5615"
                  paddingVertical={12}
                  borderRadius={8}
                  alignItems="center"
                  onPress={() => {
                    unloadModel();
                    setDetailModel(null);
                  }}
                  pressStyle={{opacity: 0.7}}>
                  <Text fontSize={15} color="#D44C56" fontWeight="600">
                    Unload Model
                  </Text>
                </YStack>
              ) : (
                <YStack
                  backgroundColor="$color9"
                  paddingVertical={12}
                  borderRadius={8}
                  alignItems="center"
                  onPress={() => {
                    if (detailModel) loadModel(detailModel.id);
                    setDetailModel(null);
                  }}
                  pressStyle={{opacity: 0.7}}>
                  <Text fontSize={15} color="white" fontWeight="600">
                    Load Model
                  </Text>
                </YStack>
              )}
            </YStack>
          </YStack>
        </YStack>
      </Modal>
    </SafeAreaView>
  );
}
