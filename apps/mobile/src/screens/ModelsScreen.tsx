import React, {useEffect, useState} from 'react';
import {
  View,
  Text,
  ScrollView,
  TouchableOpacity,
  StyleSheet,
  RefreshControl,
  ActivityIndicator,
} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {useModelStore} from '../stores/model-store';
import {StatusBadge} from '../components/StatusBadge';
import {colors, spacing, radii, typography} from '../theme';

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
            {isLoaded && (
              <StatusBadge label="Loaded" variant="success" />
            )}
          </View>
          <View style={styles.pipelineRow}>
            <View style={styles.pipelineInfo}>
              <Text style={styles.pipelineLabel}>Personality</Text>
              <Text style={styles.pipelineValue}>
                {currentSoul?.name || 'None'}
              </Text>
            </View>
          </View>
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
                    Loss: {cp.loss.toFixed(3)} · Steps: {cp.steps}
                  </Text>
                </View>
                <StatusBadge label="Use" variant="info" />
              </TouchableOpacity>
            ))}
          </View>
        )}

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Available Models</Text>
          {models.length === 0 && !loading && (
            <Text style={styles.empty}>No models found</Text>
          )}
          {models.map(model => {
            const isLoading = loadingModelId === model.id;
            return (
              <View key={model.id} style={styles.modelRow}>
                <View style={styles.modelInfo}>
                  <Text style={styles.modelName}>{model.name}</Text>
                  <Text style={styles.modelMeta}>
                    {model.size_gb
                      ? `${model.size_gb.toFixed(1)} GB`
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
              </View>
            );
          })}
        </View>
      </ScrollView>
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
});
