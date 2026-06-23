import React, {useEffect, useState, useMemo} from 'react';
import {
  View,
  Text,
  ScrollView,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  RefreshControl,
  ActivityIndicator,
  Modal,
} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {useModelStore} from '../stores/model-store';
import {StatusBadge} from '../components/StatusBadge';
import {colors, spacing, radii, typography} from '../theme';
import type {ModelInfo} from '../types';

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

  const isLoaded = health?.model_loaded;

  return (
    <SafeAreaView style={styles.safe} edges={['top']}>
      <ScrollView
        style={styles.scroll}
        contentContainerStyle={styles.content}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }>
        <Text style={styles.title}>Models</Text>

        {error && (
          <View style={styles.errorCard}>
            <Text style={styles.errorText}>{error}</Text>
            <TouchableOpacity onPress={clearError}>
              <Text style={styles.dismiss}>×</Text>
            </TouchableOpacity>
          </View>
        )}

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Active Pipeline</Text>
          <View style={styles.pipelineRow}>
            <View style={styles.pipelineInfo}>
              <Text style={styles.pipelineLabel}>Model</Text>
              <Text style={styles.pipelineValue}>
                {currentModel || 'None loaded'}
              </Text>
            </View>
            {isLoaded && <StatusBadge label="Loaded" variant="success" />}
          </View>
          <View style={styles.pipelineRow}>
            <View style={styles.pipelineInfo}>
              <Text style={styles.pipelineLabel}>Personality</Text>
              <Text style={styles.pipelineValue}>
                {currentSoul?.name || 'None'}
              </Text>
            </View>
          </View>
          {currentSoul && currentSoul.description && (
            <Text style={styles.pipelineDesc}>{currentSoul.description}</Text>
          )}
          {currentSoul && currentSoul.traits && currentSoul.traits.length > 0 && (
            <View style={styles.traitRow}>
              {currentSoul.traits.map(trait => (
                <StatusBadge key={trait} label={trait} variant="info" />
              ))}
            </View>
          )}
          {isLoaded && (
            <TouchableOpacity style={styles.unloadBtn} onPress={unloadModel}>
              <Text style={styles.unloadText}>Unload model</Text>
            </TouchableOpacity>
          )}
        </View>

        {souls.length > 0 && (
          <View style={styles.card}>
            <Text style={styles.cardTitle}>Personalities</Text>
            <ScrollView
              horizontal
              showsHorizontalScrollIndicator={false}
              contentContainerStyle={styles.chipRow}>
              {souls.map(soul => {
                const isActive = currentSoul?.name === soul.name;
                return (
                  <TouchableOpacity
                    key={soul.name}
                    style={[styles.chip, isActive && styles.chipActive]}
                    onPress={() => switchSoul(soul.name)}>
                    <Text
                      style={[styles.chipText, isActive && styles.chipTextActive]}>
                      {soul.name}
                    </Text>
                  </TouchableOpacity>
                );
              })}
            </ScrollView>
          </View>
        )}

        {checkpoints.length > 0 && (
          <View style={styles.card}>
            <Text style={styles.cardTitle}>Trained Versions</Text>
            {checkpoints.map(cp => (
              <TouchableOpacity
                key={cp.name}
                style={styles.checkpointRow}
                onPress={() => switchSoul(cp.soul, cp.name)}>
                <View style={styles.checkpointInfo}>
                  <Text style={styles.checkpointName}>{cp.name}</Text>
                  <Text style={styles.checkpointMeta}>
                    {cp.loss !== null ? `Loss: ${cp.loss.toFixed(3)}` : ''}{' '}
                    {cp.steps > 0 ? `· ${cp.steps} steps` : ''}
                  </Text>
                </View>
                <StatusBadge label="Use" variant="info" />
              </TouchableOpacity>
            ))}
          </View>
        )}

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Available Models</Text>
          {models.length > 3 && (
            <TextInput
              style={styles.searchInput}
              value={search}
              onChangeText={setSearch}
              placeholder="Search models..."
              placeholderTextColor={colors.textMuted}
              returnKeyType="search"
            />
          )}
          {filteredModels.length === 0 && !loading && (
            <Text style={styles.empty}>
              {search ? 'No models match your search' : 'No models found'}
            </Text>
          )}
          {filteredModels.map(model => {
            const isLoading = loadingModelId === model.id;
            return (
              <TouchableOpacity
                key={model.id}
                style={styles.modelRow}
                onPress={() => setDetailModel(model)}
                activeOpacity={0.7}>
                <View style={styles.modelInfo}>
                  <Text style={styles.modelName}>{model.name}</Text>
                  <Text style={styles.modelMeta}>
                    {model.size_gb
                      ? `${model.size_gb.toFixed(1)} GB`
                      : model.size_mb
                      ? `${model.size_mb} MB`
                      : model.params || model.type}
                  </Text>
                </View>
                {isLoading ? (
                  <ActivityIndicator size="small" color={colors.primary} />
                ) : model.loaded ? (
                  <StatusBadge label="Loaded" variant="success" />
                ) : (
                  <TouchableOpacity
                    style={styles.loadBtn}
                    onPress={() => loadModel(model.id)}>
                    <Text style={styles.loadBtnText}>Load</Text>
                  </TouchableOpacity>
                )}
              </TouchableOpacity>
            );
          })}
        </View>
      </ScrollView>

      {/* Model detail modal */}
      <Modal visible={!!detailModel} animationType="slide" transparent>
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <View style={styles.modalHeader}>
              <Text style={styles.modalTitle}>{detailModel?.name}</Text>
              <TouchableOpacity onPress={() => setDetailModel(null)}>
                <Text style={styles.modalClose}>×</Text>
              </TouchableOpacity>
            </View>
            <View style={styles.modalBody}>
              {detailModel?.description && (
                <Text style={styles.modalDesc}>{detailModel.description}</Text>
              )}
              <View style={styles.modalRow}>
                <Text style={styles.modalLabel}>Type</Text>
                <Text style={styles.modalValue}>{detailModel?.type || '—'}</Text>
              </View>
              <View style={styles.modalRow}>
                <Text style={styles.modalLabel}>Parameters</Text>
                <Text style={styles.modalValue}>{detailModel?.params || '—'}</Text>
              </View>
              <View style={styles.modalRow}>
                <Text style={styles.modalLabel}>Size</Text>
                <Text style={styles.modalValue}>
                  {detailModel?.size_gb
                    ? `${detailModel.size_gb.toFixed(1)} GB`
                    : detailModel?.size_mb
                    ? `${detailModel.size_mb} MB`
                    : '—'}
                </Text>
              </View>
              <View style={styles.modalRow}>
                <Text style={styles.modalLabel}>Source</Text>
                <Text style={styles.modalValue}>{detailModel?.source || '—'}</Text>
              </View>
              <View style={styles.modalRow}>
                <Text style={styles.modalLabel}>Status</Text>
                <StatusBadge
                  label={detailModel?.loaded ? 'Loaded' : 'Available'}
                  variant={detailModel?.loaded ? 'success' : 'default'}
                />
              </View>
              {detailModel?.tags && detailModel.tags.length > 0 && (
                <View style={styles.modalTags}>
                  {detailModel.tags.map(tag => (
                    <StatusBadge key={tag} label={tag} variant="info" />
                  ))}
                </View>
              )}
            </View>
            <View style={styles.modalActions}>
              {detailModel?.loaded ? (
                <TouchableOpacity
                  style={styles.modalUnloadBtn}
                  onPress={() => {
                    unloadModel();
                    setDetailModel(null);
                  }}>
                  <Text style={styles.modalUnloadText}>Unload Model</Text>
                </TouchableOpacity>
              ) : (
                <TouchableOpacity
                  style={styles.modalLoadBtn}
                  onPress={() => {
                    if (detailModel) loadModel(detailModel.id);
                    setDetailModel(null);
                  }}>
                  <Text style={styles.modalLoadText}>Load Model</Text>
                </TouchableOpacity>
              )}
            </View>
          </View>
        </View>
      </Modal>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: colors.background,
  },
  scroll: {
    flex: 1,
  },
  content: {
    padding: spacing.lg,
    gap: spacing.md,
  },
  title: {
    ...typography.h1,
    color: colors.text,
  },
  errorCard: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: '#FDE8E8',
    padding: spacing.md,
    borderRadius: radii.md,
  },
  errorText: {
    ...typography.caption,
    color: colors.error,
    flex: 1,
  },
  dismiss: {
    ...typography.body,
    color: colors.error,
    fontWeight: '600',
    marginLeft: spacing.sm,
  },
  card: {
    backgroundColor: colors.surface,
    borderRadius: radii.lg,
    padding: spacing.lg,
  },
  cardTitle: {
    ...typography.h3,
    color: colors.text,
    marginBottom: spacing.md,
  },
  pipelineRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: spacing.sm,
  },
  pipelineInfo: {
    flex: 1,
  },
  pipelineLabel: {
    ...typography.small,
    color: colors.textMuted,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  pipelineValue: {
    ...typography.body,
    color: colors.text,
    fontWeight: '500',
  },
  pipelineDesc: {
    ...typography.caption,
    color: colors.textSecondary,
    marginTop: spacing.xs,
  },
  traitRow: {
    flexDirection: 'row',
    gap: spacing.xs,
    marginTop: spacing.sm,
    flexWrap: 'wrap',
  },
  unloadBtn: {
    marginTop: spacing.md,
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.md,
    backgroundColor: colors.error + '15',
    borderRadius: radii.md,
    alignItems: 'center',
  },
  unloadText: {
    ...typography.caption,
    color: colors.error,
    fontWeight: '600',
  },
  chipRow: {
    gap: spacing.sm,
  },
  chip: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    borderRadius: radii.full,
    backgroundColor: colors.background,
    borderWidth: 1,
    borderColor: colors.border,
  },
  chipActive: {
    backgroundColor: colors.primary,
    borderColor: colors.primary,
  },
  chipText: {
    ...typography.caption,
    color: colors.textSecondary,
    fontWeight: '500',
  },
  chipTextActive: {
    color: colors.white,
  },
  checkpointRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: spacing.sm,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  checkpointInfo: {
    flex: 1,
  },
  checkpointName: {
    ...typography.body,
    color: colors.text,
    fontWeight: '500',
  },
  checkpointMeta: {
    ...typography.small,
    color: colors.textMuted,
  },
  modelRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: spacing.sm,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  modelInfo: {
    flex: 1,
  },
  modelName: {
    ...typography.body,
    color: colors.text,
    fontWeight: '500',
  },
  modelMeta: {
    ...typography.small,
    color: colors.textMuted,
  },
  loadBtn: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.xs + 2,
    borderRadius: radii.md,
    backgroundColor: colors.primary,
  },
  loadBtnText: {
    ...typography.caption,
    color: colors.white,
    fontWeight: '600',
  },
  empty: {
    ...typography.caption,
    color: colors.textMuted,
    textAlign: 'center',
    paddingVertical: spacing.lg,
  },
  searchInput: {
    ...typography.body,
    color: colors.text,
    backgroundColor: colors.background,
    borderRadius: radii.md,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    marginBottom: spacing.md,
  },
  // Modal
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.4)',
    justifyContent: 'flex-end',
  },
  modalContent: {
    backgroundColor: colors.background,
    borderTopLeftRadius: radii.xl,
    borderTopRightRadius: radii.xl,
    maxHeight: '80%',
  },
  modalHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: spacing.xl,
    paddingVertical: spacing.lg,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  modalTitle: {
    ...typography.h2,
    color: colors.text,
    flex: 1,
  },
  modalClose: {
    fontSize: 24,
    color: colors.textMuted,
    padding: spacing.xs,
  },
  modalBody: {
    paddingHorizontal: spacing.xl,
    paddingVertical: spacing.lg,
    gap: spacing.md,
  },
  modalDesc: {
    ...typography.body,
    color: colors.textSecondary,
  },
  modalRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  modalLabel: {
    ...typography.caption,
    color: colors.textMuted,
  },
  modalValue: {
    ...typography.body,
    color: colors.text,
    fontWeight: '500',
  },
  modalTags: {
    flexDirection: 'row',
    gap: spacing.xs,
    flexWrap: 'wrap',
  },
  modalActions: {
    paddingHorizontal: spacing.xl,
    paddingVertical: spacing.lg,
    borderTopWidth: 1,
    borderTopColor: colors.border,
  },
  modalLoadBtn: {
    backgroundColor: colors.primary,
    paddingVertical: spacing.md,
    borderRadius: radii.md,
    alignItems: 'center',
  },
  modalLoadText: {
    ...typography.body,
    color: colors.white,
    fontWeight: '600',
  },
  modalUnloadBtn: {
    backgroundColor: colors.error + '15',
    paddingVertical: spacing.md,
    borderRadius: radii.md,
    alignItems: 'center',
  },
  modalUnloadText: {
    ...typography.body,
    color: colors.error,
    fontWeight: '600',
  },
});
