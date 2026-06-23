import React, {useEffect, useState, useRef} from 'react';
import {
  View,
  Text,
  ScrollView,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  RefreshControl,
  ActivityIndicator,
  Alert,
  Modal,
} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {useTrainingStore, type TrainPhase} from '../stores/training-store';
import {useModelStore} from '../stores/model-store';
import {api} from '../services/api-client';
import {StatusBadge} from '../components/StatusBadge';
import {colors, spacing, radii, typography} from '../theme';

const PHASE_LABELS: Record<string, {text: string; variant: string}> = {
  idle: {text: 'Ready', variant: 'default'},
  configuring: {text: 'Configuring', variant: 'info'},
  GENERATE_DATA: {text: 'Generating Data', variant: 'info'},
  DISTILL: {text: 'Distilling', variant: 'info'},
  TRAIN: {text: 'Training', variant: 'warning'},
  TRAINING: {text: 'Training', variant: 'warning'},
  EVALUATE: {text: 'Evaluating', variant: 'info'},
  EVALUATING: {text: 'Evaluating', variant: 'info'},
  DEPLOY: {text: 'Deploying', variant: 'info'},
  COMPLETE: {text: 'Complete', variant: 'success'},
  FAILED: {text: 'Failed', variant: 'error'},
};

function LossChart({data}: {data: {step: number; value: number}[]}) {
  if (data.length < 2) {
    return (
      <View style={styles.chartPlaceholder}>
        <Text style={styles.chartPlaceholderText}>
          Loss curve will appear here
        </Text>
      </View>
    );
  }

  const maxLoss = Math.max(...data.map(d => d.value));
  const minLoss = Math.min(...data.map(d => d.value));
  const range = maxLoss - minLoss || 1;
  const W = 280;
  const H = 80;
  const pad = 4;

  return (
    <View style={styles.chartWrap}>
      <View style={[styles.chart, {width: W, height: H}]}>
        {data.map((point, i) => {
          const x = pad + (i / (data.length - 1)) * (W - pad * 2);
          const y = H - pad - ((point.value - minLoss) / range) * (H - pad * 2);
          const dotSize = i === data.length - 1 ? 6 : 3;
          const color = i === data.length - 1 ? colors.primary : colors.primaryLight;
          return (
            <View
              key={i}
              style={{
                position: 'absolute',
                left: x - dotSize / 2,
                top: y - dotSize / 2,
                width: dotSize,
                height: dotSize,
                borderRadius: dotSize / 2,
                backgroundColor: color,
              }}
            />
          );
        })}
      </View>
      <View style={styles.chartAxis}>
        <Text style={styles.chartAxisText}>{minLoss.toFixed(2)}</Text>
        <Text style={styles.chartAxisText}>{data.length} points</Text>
        <Text style={styles.chartAxisText}>{maxLoss.toFixed(2)}</Text>
      </View>
    </View>
  );
}

export function TrainingScreen() {
  const {
    phase,
    running,
    loss,
    lossHistory,
    epoch,
    totalEpochs,
    steps,
    checkpoint,
    error,
    checkpoints,
    datasets,
    config,
    setConfig,
    start,
    stop,
    refresh,
    loadCheckpoint,
    deleteCheckpoint,
    clearError,
  } = useTrainingStore();
  const modelStore = useModelStore();
  const [sourceText, setSourceText] = useState('');
  const [selectedDataset, setSelectedDataset] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [inputMode, setInputMode] = useState<'text' | 'dataset'>('text');
  const [loadingCheckpoint, setLoadingCheckpoint] = useState<string | null>(null);
  const [previewData, setPreviewData] = useState<string[]>([]);
  const [previewVisible, setPreviewVisible] = useState(false);
  const prevPhaseRef = useRef(phase);

  useEffect(() => {
    if (prevPhaseRef.current !== 'COMPLETE' && phase === 'COMPLETE') {
      Alert.alert(
        'Training Complete',
        checkpoint
          ? `Model trained successfully. Checkpoint: ${checkpoint}`
          : 'Model trained successfully.',
        [{text: 'OK'}],
      );
    }
    prevPhaseRef.current = phase;
  }, [phase, checkpoint]);

  const fetchPreview = async (datasetId: string) => {
    try {
      const result = await api.get<{rows: string[]}>(`/datasets/${datasetId}/preview`);
      setPreviewData(result.rows || []);
      setPreviewVisible(true);
    } catch {}
  };

  useEffect(() => {
    refresh();
  }, []);

  const onRefresh = async () => {
    setRefreshing(true);
    await refresh();
    setRefreshing(false);
  };

  const handleStart = () => {
    if (inputMode === 'text') {
      if (!sourceText.trim()) {
        Alert.alert('Training', 'Enter some training text first');
        return;
      }
      setConfig({source_text: sourceText, dataset_id: undefined});
    } else {
      if (!selectedDataset) {
        Alert.alert('Training', 'Select a dataset first');
        return;
      }
      setConfig({dataset_id: selectedDataset, source_text: undefined});
    }
    start();
  };

  const handleLoadCheckpoint = async (name: string) => {
    setLoadingCheckpoint(name);
    try {
      await loadCheckpoint(name);
      await modelStore.refresh();
      Alert.alert('Loaded', `Checkpoint ${name} loaded into model`);
    } catch (err: any) {
      Alert.alert('Error', err.message || 'Failed to load checkpoint');
    } finally {
      setLoadingCheckpoint(null);
    }
  };

  const isTraining = phase === 'TRAINING' || phase === 'EVALUATING' ||
    phase === 'GENERATE_DATA' || phase === 'DISTILL' || phase === 'TRAIN' ||
    phase === 'EVALUATE' || phase === 'DEPLOY';
  const isDone = phase === 'COMPLETE';
  const isFailed = phase === 'FAILED';
  const progress =
    totalEpochs > 0 ? Math.round((epoch / totalEpochs) * 100) : 0;
  const phaseInfo = PHASE_LABELS[phase] || PHASE_LABELS.idle;

  return (
    <SafeAreaView style={styles.safe} edges={['top']}>
      <ScrollView
        contentContainerStyle={styles.content}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }>
        <View style={styles.header}>
          <Text style={styles.title}>Training</Text>
          <StatusBadge
            label={phaseInfo.text}
            variant={phaseInfo.variant as any}
          />
        </View>

        {error && (
          <View style={styles.errorCard}>
            <Text style={styles.errorText}>{error}</Text>
            <TouchableOpacity onPress={clearError}>
              <Text style={styles.dismiss}>×</Text>
            </TouchableOpacity>
          </View>
        )}

        {!isTraining && !isDone && (
          <View style={styles.card}>
            <Text style={styles.cardTitle}>Training Data</Text>
            <View style={styles.modeRow}>
              <TouchableOpacity
                style={[styles.modeBtn, inputMode === 'text' && styles.modeBtnActive]}
                onPress={() => setInputMode('text')}>
                <Text style={[styles.modeBtnText, inputMode === 'text' && styles.modeBtnTextActive]}>
                  Paste Text
                </Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.modeBtn, inputMode === 'dataset' && styles.modeBtnActive]}
                onPress={() => setInputMode('dataset')}>
                <Text style={[styles.modeBtnText, inputMode === 'dataset' && styles.modeBtnTextActive]}>
                  Dataset
                </Text>
              </TouchableOpacity>
            </View>

            {inputMode === 'text' ? (
              <TextInput
                style={styles.textArea}
                value={sourceText}
                onChangeText={setSourceText}
                placeholder="Paste training text here (SRT, plain text, or lines)..."
                placeholderTextColor={colors.textMuted}
                multiline
                textAlignVertical="top"
              />
            ) : (
              <View style={styles.datasetList}>
                {datasets.length === 0 ? (
                  <Text style={styles.empty}>No datasets found</Text>
                ) : (
                  datasets.map(ds => (
                    <TouchableOpacity
                      key={ds.id}
                      style={[styles.datasetItem, selectedDataset === ds.id && styles.datasetItemActive]}
                      onPress={() => setSelectedDataset(ds.id)}>
                      <View style={styles.datasetInfo}>
                        <Text style={styles.datasetName}>{ds.name}</Text>
                        <Text style={styles.datasetMeta}>
                          {ds.file_count} files · {ds.total_chars.toLocaleString()} chars
                        </Text>
                      </View>
                      <TouchableOpacity
                        style={styles.previewBtn}
                        onPress={() => fetchPreview(ds.id)}>
                        <Text style={styles.previewBtnText}>Preview</Text>
                      </TouchableOpacity>
                      {selectedDataset === ds.id && <Text style={styles.check}>✓</Text>}
                    </TouchableOpacity>
                  ))
                )}
              </View>
            )}
          </View>
        )}

        {!isTraining && !isDone && (
          <View style={styles.card}>
            <Text style={styles.cardTitle}>Hyperparameters</Text>
            <View style={styles.paramRow}>
              <Text style={styles.paramLabel}>Epochs</Text>
              <View style={styles.paramBtns}>
                {[3, 5, 10, 20, 50].map(v => (
                  <TouchableOpacity
                    key={v}
                    style={[styles.paramBtn, config.epochs === v && styles.paramBtnActive]}
                    onPress={() => setConfig({epochs: v})}>
                    <Text style={[styles.paramBtnText, config.epochs === v && styles.paramBtnTextActive]}>{v}</Text>
                  </TouchableOpacity>
                ))}
              </View>
            </View>
            <View style={styles.paramRow}>
              <Text style={styles.paramLabel}>Learning Rate</Text>
              <View style={styles.paramBtns}>
                {[0.0001, 0.001, 0.01].map(v => (
                  <TouchableOpacity
                    key={v}
                    style={[styles.paramBtn, config.learning_rate === v && styles.paramBtnActive]}
                    onPress={() => setConfig({learning_rate: v})}>
                    <Text style={[styles.paramBtnText, config.learning_rate === v && styles.paramBtnTextActive]}>{v}</Text>
                  </TouchableOpacity>
                ))}
              </View>
            </View>
            <View style={styles.paramRow}>
              <Text style={styles.paramLabel}>Soul</Text>
              <View style={styles.paramBtns}>
                {['assistant', 'creative', 'coder', 'teacher', 'analyst'].map(v => (
                  <TouchableOpacity
                    key={v}
                    style={[styles.paramBtn, config.soul_name === v && styles.paramBtnActive]}
                    onPress={() => setConfig({soul_name: v})}>
                    <Text style={[styles.paramBtnText, config.soul_name === v && styles.paramBtnTextActive]}>{v}</Text>
                  </TouchableOpacity>
                ))}
              </View>
            </View>
          </View>
        )}

        {(isTraining || isDone || isFailed) && (
          <View style={styles.card}>
            <Text style={styles.cardTitle}>Progress</Text>
            <View style={styles.progressHeader}>
              <Text style={styles.progressText}>Epoch {epoch}/{totalEpochs}</Text>
              <Text style={styles.progressText}>{progress}%</Text>
            </View>
            <View style={styles.progressTrack}>
              <View
                style={[
                  styles.progressFill,
                  {
                    width: `${progress}%`,
                    backgroundColor: isDone ? colors.success : isFailed ? colors.error : colors.primary,
                  },
                ]}
              />
            </View>
            <View style={styles.statsRow}>
              <View style={styles.stat}>
                <Text style={styles.statLabel}>Loss</Text>
                <Text style={styles.statValue}>{loss !== null ? loss.toFixed(4) : '—'}</Text>
              </View>
              <View style={styles.stat}>
                <Text style={styles.statLabel}>Steps</Text>
                <Text style={styles.statValue}>{steps}</Text>
              </View>
            </View>
            <LossChart data={lossHistory} />
            {isTraining && (
              <View style={styles.progressActions}>
                <ActivityIndicator size="small" color={colors.primary} />
                <Text style={styles.trainingText}>{phaseInfo.text}...</Text>
              </View>
            )}
          </View>
        )}

        {isDone && checkpoint && (
          <View style={styles.card}>
            <Text style={styles.cardTitle}>Training Complete</Text>
            <StatusBadge label="Success" variant="success" />
            <Text style={styles.completeText}>Checkpoint: {checkpoint}</Text>
            <Text style={styles.completeMeta}>
              Final loss: {loss?.toFixed(4) || '—'} · {steps} steps
            </Text>
            <TouchableOpacity
              style={styles.loadBtn}
              onPress={() => handleLoadCheckpoint(checkpoint)}
              disabled={loadingCheckpoint === checkpoint}>
              {loadingCheckpoint === checkpoint ? (
                <ActivityIndicator size="small" color={colors.white} />
              ) : (
                <Text style={styles.loadBtnText}>Load Model for Chat</Text>
              )}
            </TouchableOpacity>
          </View>
        )}

        <View style={styles.actions}>
          {isTraining ? (
            <TouchableOpacity style={styles.stopBtn} onPress={stop}>
              <Text style={styles.stopBtnText}>Stop Training</Text>
            </TouchableOpacity>
          ) : (
            <TouchableOpacity
              style={[styles.startBtn, running && styles.startBtnDisabled]}
              onPress={handleStart}
              disabled={running}>
              <Text style={styles.startBtnText}>
                {isDone ? 'Train Again' : 'Start Training'}
              </Text>
            </TouchableOpacity>
          )}
        </View>

        {checkpoints.length > 0 && (
          <View style={styles.card}>
            <Text style={styles.cardTitle}>Checkpoints</Text>
            {checkpoints.map(cp => (
              <View key={cp.name} style={styles.ckptRow}>
                <View style={styles.ckptInfo}>
                  <Text style={styles.ckptName}>{cp.name}</Text>
                  <Text style={styles.ckptMeta}>
                    {cp.loss !== null ? `Loss: ${cp.loss.toFixed(3)}` : ''}{' '}
                    {cp.steps > 0 ? `· ${cp.steps} steps` : ''}{' '}
                    {cp.size_mb ? `· ${cp.size_mb} MB` : ''}
                  </Text>
                  <View style={styles.ckptBadges}>
                    {cp.soul && cp.soul !== 'unknown' && (
                      <StatusBadge label={cp.soul} variant="info" />
                    )}
                    {cp.verdict && (
                      <StatusBadge
                        label={cp.verdict}
                        variant={cp.verdict === 'improved' ? 'success' : 'warning'}
                      />
                    )}
                  </View>
                </View>
                <View style={styles.ckptActions}>
                  <TouchableOpacity
                    style={styles.ckptLoad}
                    onPress={() => handleLoadCheckpoint(cp.name)}
                    disabled={loadingCheckpoint === cp.name}>
                    {loadingCheckpoint === cp.name ? (
                      <ActivityIndicator size="small" color={colors.primary} />
                    ) : (
                      <Text style={styles.ckptLoadText}>Load</Text>
                    )}
                  </TouchableOpacity>
                  <TouchableOpacity
                    style={styles.ckptDelete}
                    onPress={() => {
                      Alert.alert('Delete', `Delete ${cp.name}?`, [
                        {text: 'Cancel', style: 'cancel'},
                        {text: 'Delete', style: 'destructive', onPress: () => deleteCheckpoint(cp.name)},
                      ]);
                    }}>
                    <Text style={styles.ckptDeleteText}>×</Text>
                  </TouchableOpacity>
                </View>
              </View>
            ))}
          </View>
        )}
      </ScrollView>

      {/* Dataset preview modal */}
      <Modal visible={previewVisible} animationType="slide" transparent>
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <View style={styles.modalHeader}>
              <Text style={styles.modalTitle}>Dataset Preview</Text>
              <TouchableOpacity onPress={() => setPreviewVisible(false)}>
                <Text style={styles.modalClose}>×</Text>
              </TouchableOpacity>
            </View>
            <ScrollView style={styles.modalBody}>
              {previewData.map((line, i) => (
                <View key={i} style={styles.previewLine}>
                  <Text style={styles.previewNum}>{i + 1}</Text>
                  <Text style={styles.previewText} numberOfLines={3}>{line}</Text>
                </View>
              ))}
              {previewData.length === 0 && (
                <Text style={styles.previewEmpty}>No preview available</Text>
              )}
            </ScrollView>
          </View>
        </View>
      </Modal>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: {flex: 1, backgroundColor: colors.background},
  content: {padding: spacing.lg, gap: spacing.md},
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  title: {...typography.h1, color: colors.text},
  errorCard: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: '#FDE8E8',
    padding: spacing.md,
    borderRadius: radii.md,
  },
  errorText: {...typography.caption, color: colors.error, flex: 1},
  dismiss: {...typography.body, color: colors.error, fontWeight: '600'},
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
  modeRow: {
    flexDirection: 'row',
    gap: spacing.sm,
    marginBottom: spacing.md,
  },
  modeBtn: {
    flex: 1,
    paddingVertical: spacing.sm,
    borderRadius: radii.md,
    backgroundColor: colors.background,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: colors.border,
  },
  modeBtnActive: {
    backgroundColor: colors.primary,
    borderColor: colors.primary,
  },
  modeBtnText: {...typography.caption, color: colors.textSecondary, fontWeight: '500'},
  modeBtnTextActive: {color: colors.white},
  textArea: {
    ...typography.body,
    color: colors.text,
    backgroundColor: colors.background,
    borderRadius: radii.md,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.md,
    minHeight: 120,
  },
  datasetList: {gap: spacing.xs},
  datasetItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: spacing.md,
    backgroundColor: colors.background,
    borderRadius: radii.md,
    borderWidth: 1,
    borderColor: colors.border,
  },
  datasetItemActive: {
    borderColor: colors.primary,
    backgroundColor: colors.primary + '10',
  },
  datasetInfo: {flex: 1},
  datasetName: {...typography.body, color: colors.text, fontWeight: '500'},
  datasetMeta: {...typography.small, color: colors.textMuted},
  check: {color: colors.primary, fontSize: 18, fontWeight: '700'},
  empty: {...typography.caption, color: colors.textMuted, textAlign: 'center', padding: spacing.lg},
  paramRow: {marginBottom: spacing.md},
  paramLabel: {...typography.caption, color: colors.textSecondary, marginBottom: spacing.xs},
  paramBtns: {flexDirection: 'row', gap: spacing.xs, flexWrap: 'wrap'},
  paramBtn: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.xs,
    borderRadius: radii.full,
    backgroundColor: colors.background,
    borderWidth: 1,
    borderColor: colors.border,
  },
  paramBtnActive: {backgroundColor: colors.primary, borderColor: colors.primary},
  paramBtnText: {...typography.small, color: colors.textSecondary},
  paramBtnTextActive: {color: colors.white},
  progressHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: spacing.sm,
  },
  progressText: {...typography.body, color: colors.text, fontWeight: '500'},
  progressTrack: {
    height: 8,
    backgroundColor: colors.border,
    borderRadius: 4,
    overflow: 'hidden',
    marginBottom: spacing.md,
  },
  progressFill: {height: '100%', borderRadius: 4},
  statsRow: {
    flexDirection: 'row',
    gap: spacing.xl,
    marginBottom: spacing.md,
  },
  stat: {},
  statLabel: {...typography.small, color: colors.textMuted},
  statValue: {...typography.h3, color: colors.text},
  chartWrap: {marginTop: spacing.sm},
  chart: {
    backgroundColor: colors.background,
    borderRadius: radii.sm,
    overflow: 'hidden',
  },
  chartPlaceholder: {
    height: 80,
    backgroundColor: colors.background,
    borderRadius: radii.sm,
    alignItems: 'center',
    justifyContent: 'center',
  },
  chartPlaceholderText: {...typography.caption, color: colors.textMuted},
  chartAxis: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: spacing.xs,
  },
  chartAxisText: {...typography.small, color: colors.textMuted},
  progressActions: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    marginTop: spacing.md,
  },
  trainingText: {...typography.caption, color: colors.textMuted},
  completeText: {...typography.body, color: colors.text, marginTop: spacing.md},
  completeMeta: {...typography.small, color: colors.textMuted, marginTop: spacing.xs},
  loadBtn: {
    marginTop: spacing.md,
    backgroundColor: colors.primary,
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.lg,
    borderRadius: radii.md,
    alignItems: 'center',
  },
  loadBtnText: {...typography.caption, color: colors.white, fontWeight: '600'},
  actions: {gap: spacing.sm},
  startBtn: {
    backgroundColor: colors.primary,
    paddingVertical: spacing.md,
    borderRadius: radii.md,
    alignItems: 'center',
  },
  startBtnDisabled: {opacity: 0.5},
  startBtnText: {...typography.body, color: colors.white, fontWeight: '600'},
  stopBtn: {
    backgroundColor: colors.error,
    paddingVertical: spacing.md,
    borderRadius: radii.md,
    alignItems: 'center',
  },
  stopBtnText: {...typography.body, color: colors.white, fontWeight: '600'},
  ckptRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: spacing.sm,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  ckptInfo: {flex: 1, gap: spacing.xs},
  ckptName: {...typography.body, color: colors.text, fontWeight: '500'},
  ckptMeta: {...typography.small, color: colors.textMuted},
  ckptBadges: {flexDirection: 'row', gap: spacing.xs},
  ckptActions: {flexDirection: 'row', gap: spacing.xs, alignItems: 'center'},
  ckptLoad: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.xs,
    borderRadius: radii.md,
    backgroundColor: colors.primary + '15',
  },
  ckptLoadText: {...typography.small, color: colors.primary, fontWeight: '600'},
  ckptDelete: {
    width: 28,
    height: 28,
    borderRadius: radii.full,
    backgroundColor: colors.error + '15',
    alignItems: 'center',
    justifyContent: 'center',
  },
  ckptDeleteText: {color: colors.error, fontSize: 16, fontWeight: '600'},
  previewBtn: {
    paddingHorizontal: spacing.sm,
    paddingVertical: 2,
    borderRadius: radii.sm,
    backgroundColor: colors.primary + '15',
  },
  previewBtnText: {
    ...typography.small,
    color: colors.primary,
    fontWeight: '500',
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.4)',
    justifyContent: 'flex-end',
  },
  modalContent: {
    backgroundColor: colors.background,
    borderTopLeftRadius: radii.xl,
    borderTopRightRadius: radii.xl,
    maxHeight: '70%',
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
    ...typography.h3,
    color: colors.text,
  },
  modalClose: {
    fontSize: 24,
    color: colors.textMuted,
    padding: spacing.xs,
  },
  modalBody: {
    paddingHorizontal: spacing.xl,
    paddingVertical: spacing.md,
  },
  previewLine: {
    flexDirection: 'row',
    gap: spacing.sm,
    paddingVertical: spacing.xs,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  previewNum: {
    ...typography.small,
    color: colors.textMuted,
    width: 24,
  },
  previewText: {
    ...typography.caption,
    color: colors.text,
    flex: 1,
  },
  previewEmpty: {
    ...typography.caption,
    color: colors.textMuted,
    textAlign: 'center',
    padding: spacing.xxl,
  },
});
