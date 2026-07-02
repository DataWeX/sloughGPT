import React, {useEffect, useState, useRef} from 'react';
import {
  View,
  Text,
  ScrollView,
  TouchableOpacity,
  StyleSheet,
  RefreshControl,
  ActivityIndicator,
  Alert,
} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {useActivityStore} from '../stores/activity-store';
import {useSensor} from '../hooks/useSensor';
import {SensorGraph} from '../components/SensorGraph';
import {StatusBadge} from '../components/StatusBadge';
import {colors, spacing, radii, typography} from '../theme';
import {ACTIVITY_NAMES} from '../types';
import type {ActivityPrediction} from '../types';
import type {LossPoint} from '../stores/activity-store';

const WINDOW_SIZE = 128;

export function ActivityScreen() {
  const {
    sensorHistory,
    isRecording,
    recordingLabel,
    phase,
    trainingEpochs,
    trainingEpoch,
    trainingLoss,
    trainingAccuracy,
    lossHistory,
    numSamples,
    error,
    modelLoaded,
    dataset,
    totalRecordings,
    bgBufferSize,
    bgLastSync,
    lastPrediction,
    clearHistory,
    startRecording,
    stopRecording,
    setRecordingLabel,
    startTraining,
    startTrainingStream,
    predictActivity,
    refreshStatus,
    clearError,
    deleteAll,
    setBackgroundState,
    setModelSync,
  } = useActivityStore();

  const [refreshing, setRefreshing] = useState(false);
  const [trainingOpts, setTrainingOpts] = useState<{epochs: number}>({
    epochs: 30,
  });
  const [predicting, setPredicting] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [bgActive, setBgActive] = useState(false);
  const [autoTrainActive, setAutoTrainActive] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Start sensor when recording
  useSensor(isRecording);

  // Auto-predict when recording stops
  const prevRecordingRef = useRef(isRecording);
  useEffect(() => {
    if (prevRecordingRef.current && !isRecording) {
      const window = sensorHistory.slice(-WINDOW_SIZE);
      if (window.length >= 10 && modelLoaded) {
        const data = window.map(r => [
          r.accel.x, r.accel.y, r.accel.z,
          r.gyro.x, r.gyro.y, r.gyro.z,
        ]);
        setPredicting(true);
        predictActivity(data).finally(() => setPredicting(false));
      }
    }
    prevRecordingRef.current = isRecording;
  }, [isRecording]);

  // Auto-refresh status
  useEffect(() => {
    refreshStatus();
    intervalRef.current = setInterval(refreshStatus, 10000);
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);

  // Background recording toggle
  const handleToggleBackground = async () => {
    if (bgActive) {
      const {stopBackgroundRecording} = require('../services/background-recorder');
      await stopBackgroundRecording();
      setBgActive(false);
      setBackgroundState(false);
    } else {
      const {startBackgroundRecording, getBufferSize, getLastSyncTime} =
        require('../services/background-recorder');
      await startBackgroundRecording();
      setBgActive(true);
      const size = await getBufferSize();
      const lastSync = await getLastSyncTime();
      setBackgroundState(true, size, lastSync);
    }
  };

  // Auto-train toggle
  const handleToggleAutoTrain = async () => {
    if (autoTrainActive) {
      const {stopAutoTrainScheduler} = require('../services/auto-train-scheduler');
      stopAutoTrainScheduler();
      setAutoTrainActive(false);
    } else {
      const {startAutoTrainScheduler} = require('../services/auto-train-scheduler');
      startAutoTrainScheduler();
      setAutoTrainActive(true);
    }
  };

  // Model sync after training completes
  useEffect(() => {
    if (phase === 'complete') {
      const {syncModel, getModelSyncStatus} = require('../services/model-sync');
      syncModel().then(() => getModelSyncStatus().then(setModelSync));
    }
  }, [phase]);

  const onRefresh = async () => {
    setRefreshing(true);
    await refreshStatus();
    setRefreshing(false);
  };

  const handleStartRecording = () => {
    startRecording(recordingLabel ?? undefined);
  };

  const handleStopRecording = async () => {
    stopRecording();
  };

  const handleSaveRecording = async () => {
    const window = sensorHistory.slice(-WINDOW_SIZE);
    if (window.length < 10) {
      Alert.alert('Not enough data', 'Record at least 10 samples (~1 second).');
      return;
    }
    const data = window.map(r => [
      r.accel.x, r.accel.y, r.accel.z,
      r.gyro.x, r.gyro.y, r.gyro.z,
    ]);
    try {
      const {recordData} = require('../services/activity-service');
      await recordData({
        data,
        label: recordingLabel ?? undefined,
      });
      Alert.alert('Saved', `${data.length} samples recorded.`);
      await refreshStatus();
    } catch (err: any) {
      Alert.alert('Error', err.message || 'Failed to save recording');
    }
  };

  const handleTrain = async () => {
    if (dataset.length < 5) {
      Alert.alert(
        'Need more data',
        `Have ${dataset.length} recordings, need at least 5 labeled recordings.`,
      );
      return;
    }
    Alert.alert(
      'Start Training',
      `Train for ${trainingOpts.epochs} epochs on ${totalRecordings} recordings?\n\nUses SSE streaming — live loss updates.`,
      [
        {text: 'Cancel', style: 'cancel'},
        {
          text: 'Train',
          onPress: () => startTrainingStream(trainingOpts),
        },
      ],
    );
  };

  const handlePredict = async () => {
    const window = sensorHistory.slice(-WINDOW_SIZE);
    if (window.length < 10) {
      Alert.alert('No data', 'Record some sensor data first.');
      return;
    }
    const data = window.map(r => [
      r.accel.x, r.accel.y, r.accel.z,
      r.gyro.x, r.gyro.y, r.gyro.z,
    ]);
    setPredicting(true);
    await predictActivity(data);
    setPredicting(false);
  };

  const handleDeleteAll = () => {
    Alert.alert(
      'Delete All Data',
      `Delete ${totalRecordings} recordings and reset model?`,
      [
        {text: 'Cancel', style: 'cancel'},
        {
          text: 'Delete',
          style: 'destructive',
          onPress: async () => {
            setDeleting(true);
            await deleteAll();
            setDeleting(false);
          },
        },
      ],
    );
  };

  const canRecord = !isRecording;
  const canPredict = modelLoaded && sensorHistory.length >= 10;
  const labeledCount = dataset.filter(r => r.label >= 0).length;

  return (
    <SafeAreaView style={styles.safe} edges={['top']}>
      <ScrollView
        contentContainerStyle={styles.content}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }>
        {/* Header */}
        <View style={styles.header}>
          <View>
            <Text style={styles.title}>Activity</Text>
            <Text style={styles.subtitle}>
              {isRecording ? 'Recording sensor data...' : 'Collect and classify'}
            </Text>
          </View>
          <StatusBadge
            label={modelLoaded ? 'Model Ready' : 'No Model'}
            variant={modelLoaded ? 'success' : 'default'}
          />
        </View>

        {/* Error */}
        {error && (
          <View style={styles.errorCard}>
            <Text style={styles.errorText}>{error}</Text>
            <TouchableOpacity onPress={clearError}>
              <Text style={styles.dismiss}>×</Text>
            </TouchableOpacity>
          </View>
        )}

        {/* Live Sensor Graph */}
        {sensorHistory.length > 0 && (
          <View style={styles.card}>
            <View style={styles.sensorHeader}>
              <Text style={styles.cardTitle}>
                {isRecording ? 'Live Sensor' : 'Latest Window'}
              </Text>
              <Text style={styles.sensorCount}>
                {sensorHistory.length} samples
              </Text>
            </View>
            <SensorGraph
              data={sensorHistory.slice(-WINDOW_SIZE)}
              channels={['accel_x', 'accel_y', 'accel_z']}
              width={300}
              height={80}
            />
            <SensorGraph
              data={sensorHistory.slice(-WINDOW_SIZE)}
              channels={['gyro_x', 'gyro_y', 'gyro_z']}
              width={300}
              height={80}
            />
          </View>
        )}

        {/* Recording Controls */}
        <View style={styles.card}>
          <Text style={styles.cardTitle}>
            {isRecording ? 'Recording...' : 'Record Sensor Data'}
          </Text>

          {/* Activity label picker */}
          <View style={styles.labelRow}>
            <Text style={styles.labelText}>Activity:</Text>
            <ScrollView horizontal showsHorizontalScrollIndicator={false}>
              <View style={styles.labelChips}>
                <TouchableOpacity
                  style={[
                    styles.chip,
                    recordingLabel === null && styles.chipActive,
                  ]}
                  onPress={() => setRecordingLabel(null)}>
                  <Text
                    style={[
                      styles.chipText,
                      recordingLabel === null && styles.chipTextActive,
                    ]}>
                    Auto
                  </Text>
                </TouchableOpacity>
                {ACTIVITY_NAMES.map((name, i) => (
                  <TouchableOpacity
                    key={name}
                    style={[
                      styles.chip,
                      recordingLabel === i && styles.chipActive,
                    ]}
                    onPress={() => setRecordingLabel(i)}>
                    <Text
                      style={[
                        styles.chipText,
                        recordingLabel === i && styles.chipTextActive,
                      ]}>
                      {name}
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>
            </ScrollView>
          </View>

          {/* Record / Stop / Save buttons */}
          <View style={styles.recordActions}>
            {isRecording ? (
              <TouchableOpacity
                style={styles.stopBtn}
                onPress={handleStopRecording}>
                <View style={styles.stopIcon} />
                <Text style={styles.stopBtnText}>Stop</Text>
              </TouchableOpacity>
            ) : (
              <TouchableOpacity
                style={styles.recordBtn}
                onPress={handleStartRecording}>
                <View style={styles.recordIcon} />
                <Text style={styles.recordBtnText}>Record</Text>
              </TouchableOpacity>
            )}

            {!isRecording && sensorHistory.length >= 10 && (
              <TouchableOpacity
                style={styles.saveBtn}
                onPress={handleSaveRecording}>
                <Text style={styles.saveBtnText}>Save</Text>
              </TouchableOpacity>
            )}

            <TouchableOpacity style={styles.clearBtn} onPress={clearHistory}>
              <Text style={styles.clearBtnText}>Clear</Text>
            </TouchableOpacity>
          </View>

          {/* Background recording toggle */}
          <View style={styles.bgRow}>
            <TouchableOpacity
              style={[styles.bgToggle, bgActive && styles.bgToggleActive]}
              onPress={handleToggleBackground}>
              <Text style={styles.bgToggleText}>
                {bgActive ? '● Background ON' : '○ Background OFF'}
              </Text>
            </TouchableOpacity>
            {bgActive && (
              <Text style={styles.bgMeta}>
                buffer: {bgBufferSize} · last sync: {bgLastSync ? new Date(bgLastSync).toLocaleTimeString() : 'never'}
              </Text>
            )}
          </View>
        </View>

        {/* Prediction */}
        {lastPrediction && (
          <View style={styles.card}>
            <View style={styles.predictionHeader}>
              <Text style={styles.cardTitle}>Latest Prediction</Text>
              <StatusBadge
                label={lastPrediction.activity}
                variant="info"
              />
            </View>
            <Text style={styles.confidenceText}>
              Confidence: {(lastPrediction.confidence * 100).toFixed(1)}%
            </Text>
            <View style={styles.probBar}>
              {lastPrediction.probabilities.map((p, i) => (
                <View
                  key={i}
                  style={[
                    styles.probSegment,
                    {
                      flex: p,
                      backgroundColor:
                        i === lastPrediction.class_id
                          ? colors.primary
                          : colors.primaryLight + '50',
                    },
                  ]}
                />
              ))}
            </View>
            <View style={styles.probLabels}>
              {ACTIVITY_NAMES.map((name, i) => (
                <Text
                  key={name}
                  style={[
                    styles.probLabel,
                    i === lastPrediction.class_id && styles.probLabelActive,
                  ]}
                  numberOfLines={1}>
                  {name}
                </Text>
              ))}
            </View>
          </View>
        )}

        {/* Predict button */}
        {!isRecording && (
          <TouchableOpacity
            style={[styles.predictBtn, !canPredict && styles.predictBtnDisabled]}
            onPress={handlePredict}
            disabled={!canPredict || predicting}>
            {predicting ? (
              <ActivityIndicator size="small" color={colors.white} />
            ) : (
              <Text style={styles.predictBtnText}>Predict Current Window</Text>
            )}
          </TouchableOpacity>
        )}

        {/* Training Section */}
        <View style={styles.card}>
          <View style={styles.trainHeader}>
            <Text style={styles.cardTitle}>Training</Text>
            <StatusBadge
              label={
                phase === 'training'
                  ? 'Running'
                  : phase === 'complete'
                  ? 'Done'
                  : phase === 'failed'
                  ? 'Failed'
                  : 'Idle'
              }
              variant={
                phase === 'training'
                  ? 'warning'
                  : phase === 'complete'
                  ? 'success'
                  : phase === 'failed'
                  ? 'error'
                  : 'default'
              }
            />
          </View>

          <View style={styles.statsRow}>
            <View style={styles.stat}>
              <Text style={styles.statLabel}>Recordings</Text>
              <Text style={styles.statValue}>{totalRecordings}</Text>
            </View>
            <View style={styles.stat}>
              <Text style={styles.statLabel}>Labeled</Text>
              <Text style={styles.statValue}>{labeledCount}</Text>
            </View>
            <View style={styles.stat}>
              <Text style={styles.statLabel}>Model</Text>
              <StatusBadge
                label={modelLoaded ? 'Yes' : 'No'}
                variant={modelLoaded ? 'success' : 'default'}
              />
            </View>
          </View>

          {phase === 'complete' && (
            <View style={styles.trainResult}>
              <Text style={styles.resultTitle}>Training Complete</Text>
              <View style={styles.resultRow}>
                <Text style={styles.resultLabel}>Accuracy</Text>
                <Text style={styles.resultValue}>
                  {trainingAccuracy !== null
                    ? `${(trainingAccuracy * 100).toFixed(1)}%`
                    : '—'}
                </Text>
              </View>
              <View style={styles.resultRow}>
                <Text style={styles.resultLabel}>Loss</Text>
                <Text style={styles.resultValue}>
                  {trainingLoss !== null ? trainingLoss.toFixed(4) : '—'}
                </Text>
              </View>
              <View style={styles.resultRow}>
                <Text style={styles.resultLabel}>Samples</Text>
                <Text style={styles.resultValue}>{numSamples}</Text>
              </View>
            </View>
          )}

          {phase === 'failed' && error && (
            <View style={styles.trainError}>
              <Text style={styles.trainErrorText}>{error}</Text>
            </View>
          )}

          {/* Epoch selector */}
          {phase !== 'training' && (
            <View style={styles.epochRow}>
              <Text style={styles.epochLabel}>Epochs</Text>
              <View style={styles.epochBtns}>
                {[10, 30, 60, 100].map(v => (
                  <TouchableOpacity
                    key={v}
                    style={[
                      styles.epochBtn,
                      trainingOpts.epochs === v && styles.epochBtnActive,
                    ]}
                    onPress={() => setTrainingOpts(s => ({...s, epochs: v}))}>
                    <Text
                      style={[
                        styles.epochBtnText,
                        trainingOpts.epochs === v && styles.epochBtnTextActive,
                      ]}>
                      {v}
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>
            </View>
          )}

          {phase === 'training' ? (
            <View style={styles.trainingSpinner}>
              <ActivityIndicator size="small" color={colors.primary} />
              <Text style={styles.trainingText}>Training classifier...</Text>
            </View>
          ) : (
            <TouchableOpacity
              style={styles.trainBtn}
              onPress={handleTrain}
              disabled={labeledCount < 2}>
              <Text style={styles.trainBtnText}>
                Train Classifier ({trainingOpts.epochs} epochs)
              </Text>
            </TouchableOpacity>
          )}

          {/* Loss chart during training */}
          {phase === 'training' && lossHistory.length > 1 && (
            <View style={styles.lossChart}>
              <Text style={styles.lossChartTitle}>Loss Curve</Text>
              <View style={styles.lossChartInner}>
                {lossHistory.map((pt: LossPoint, i: number) => {
                  const maxLoss = Math.max(...lossHistory.map(l => l.loss), 0.1);
                  const w = 280;
                  const h = 60;
                  const x = 4 + (i / (lossHistory.length - 1)) * (w - 8);
                  const y = 4 + (1 - pt.loss / maxLoss) * (h - 8);
                  return (
                    <View key={i} style={{
                      position: 'absolute', left: x - 2, top: y - 2,
                      width: 4, height: 4, borderRadius: 2,
                      backgroundColor: colors.accent,
                    }} />
                  );
                })}
              </View>
              <View style={styles.lossChartLabels}>
                <Text style={styles.lossChartLabel}>epoch {trainingEpoch}/{trainingEpochs}</Text>
                <Text style={styles.lossChartLabel}>loss: {trainingLoss?.toFixed(3) ?? '—'}</Text>
              </View>
            </View>
          )}

          {/* Auto-train toggle */}
          <View style={styles.bgRow}>
            <TouchableOpacity
              style={[styles.bgToggle, autoTrainActive && styles.bgToggleActive]}
              onPress={handleToggleAutoTrain}>
              <Text style={styles.bgToggleText}>
                {autoTrainActive ? '● Auto-train ON' : '○ Auto-train OFF'}
              </Text>
            </TouchableOpacity>
            <Text style={styles.bgMeta}>
              triggers when ≥10 new recordings accumulated
            </Text>
          </View>
        </View>

        {/* Dataset stats */}
        <View style={styles.card}>
          <View style={styles.dsHeader}>
            <Text style={styles.cardTitle}>Dataset</Text>
            <Text style={styles.dsCount}>{totalRecordings} recordings</Text>
          </View>
          {dataset.length === 0 ? (
            <Text style={styles.empty}>No recordings yet. Record some sensor data.</Text>
          ) : (
            dataset.slice(-10).reverse().map(r => (
              <View key={r.id} style={styles.recordingRow}>
                <View style={styles.recInfo}>
                  <StatusBadge
                    label={r.activity}
                    variant={
                      r.label >= 0 ? 'success' : 'default'
                    }
                  />
                  <Text style={styles.recMeta}>
                    {r.samples} samples · #{r.id}
                  </Text>
                </View>
              </View>
            ))
          )}
          {dataset.length > 10 && (
            <Text style={styles.moreText}>
              +{dataset.length - 10} older recordings
            </Text>
          )}
          {totalRecordings > 0 && (
            <TouchableOpacity
              style={styles.deleteAllBtn}
              onPress={handleDeleteAll}
              disabled={deleting}>
              {deleting ? (
                <ActivityIndicator size="small" color={colors.error} />
              ) : (
                <Text style={styles.deleteAllText}>Delete All Data</Text>
              )}
            </TouchableOpacity>
          )}
        </View>
      </ScrollView>
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
  subtitle: {...typography.caption, color: colors.textMuted, marginTop: 2},
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
  sensorHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: spacing.sm,
  },
  sensorCount: {...typography.small, color: colors.textMuted},
  labelRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    marginBottom: spacing.md,
  },
  labelText: {...typography.caption, color: colors.textSecondary, fontWeight: '500'},
  labelChips: {flexDirection: 'row', gap: spacing.xs},
  chip: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.xs,
    borderRadius: radii.full,
    backgroundColor: colors.background,
    borderWidth: 1,
    borderColor: colors.border,
  },
  chipActive: {backgroundColor: colors.primary, borderColor: colors.primary},
  chipText: {...typography.small, color: colors.textSecondary},
  chipTextActive: {color: colors.white},
  recordActions: {
    flexDirection: 'row',
    gap: spacing.sm,
    alignItems: 'center',
  },
  recordBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    backgroundColor: colors.error,
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.lg,
    borderRadius: radii.md,
  },
  recordIcon: {
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: colors.white,
  },
  recordBtnText: {...typography.caption, color: colors.white, fontWeight: '600'},
  stopBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    backgroundColor: colors.textSecondary,
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.lg,
    borderRadius: radii.md,
  },
  stopIcon: {
    width: 12,
    height: 12,
    borderRadius: 2,
    backgroundColor: colors.white,
  },
  stopBtnText: {...typography.caption, color: colors.white, fontWeight: '600'},
  saveBtn: {
    backgroundColor: colors.success,
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.lg,
    borderRadius: radii.md,
  },
  saveBtnText: {...typography.caption, color: colors.white, fontWeight: '600'},
  clearBtn: {
    backgroundColor: colors.background,
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.lg,
    borderRadius: radii.md,
    borderWidth: 1,
    borderColor: colors.border,
  },
  clearBtnText: {...typography.caption, color: colors.textSecondary, fontWeight: '500'},
  predictionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  confidenceText: {
    ...typography.body,
    color: colors.textSecondary,
    marginBottom: spacing.sm,
  },
  probBar: {
    flexDirection: 'row',
    height: 6,
    borderRadius: 3,
    overflow: 'hidden',
    marginBottom: spacing.xs,
  },
  probSegment: {borderRadius: 0},
  probLabels: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  probLabel: {
    ...typography.small,
    color: colors.textMuted,
    flex: 1,
    textAlign: 'center',
  },
  probLabelActive: {
    color: colors.primary,
    fontWeight: '600',
  },
  predictBtn: {
    backgroundColor: colors.primary,
    paddingVertical: spacing.md,
    borderRadius: radii.md,
    alignItems: 'center',
  },
  predictBtnDisabled: {opacity: 0.5},
  predictBtnText: {...typography.body, color: colors.white, fontWeight: '600'},
  trainHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  statsRow: {
    flexDirection: 'row',
    gap: spacing.lg,
    marginBottom: spacing.md,
  },
  stat: {alignItems: 'center'},
  statLabel: {...typography.small, color: colors.textMuted},
  statValue: {...typography.h3, color: colors.text, marginTop: 2},
  trainResult: {
    backgroundColor: '#E8F5EE',
    borderRadius: radii.md,
    padding: spacing.md,
    marginBottom: spacing.md,
  },
  resultTitle: {
    ...typography.body,
    color: colors.success,
    fontWeight: '600',
    marginBottom: spacing.sm,
  },
  resultRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 2,
  },
  resultLabel: {...typography.caption, color: colors.textSecondary},
  resultValue: {...typography.caption, color: colors.text, fontWeight: '500'},
  trainError: {
    backgroundColor: '#FDE8E8',
    borderRadius: radii.md,
    padding: spacing.md,
    marginBottom: spacing.md,
  },
  trainErrorText: {...typography.caption, color: colors.error},
  epochRow: {marginBottom: spacing.md},
  epochLabel: {
    ...typography.caption,
    color: colors.textSecondary,
    marginBottom: spacing.xs,
  },
  epochBtns: {flexDirection: 'row', gap: spacing.xs},
  epochBtn: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.xs,
    borderRadius: radii.full,
    backgroundColor: colors.background,
    borderWidth: 1,
    borderColor: colors.border,
  },
  epochBtnActive: {
    backgroundColor: colors.primary,
    borderColor: colors.primary,
  },
  epochBtnText: {...typography.small, color: colors.textSecondary},
  epochBtnTextActive: {color: colors.white},
  trainingSpinner: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    paddingVertical: spacing.sm,
  },
  trainingText: {...typography.caption, color: colors.textMuted},
  trainBtn: {
    backgroundColor: colors.accent,
    paddingVertical: spacing.sm,
    borderRadius: radii.md,
    alignItems: 'center',
  },
  trainBtnText: {...typography.caption, color: colors.white, fontWeight: '600'},
  dsHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: spacing.sm,
  },
  dsCount: {...typography.small, color: colors.textMuted},
  empty: {
    ...typography.caption,
    color: colors.textMuted,
    textAlign: 'center',
    paddingVertical: spacing.lg,
  },
  recordingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: spacing.sm,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  recInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  recMeta: {...typography.small, color: colors.textMuted},
  moreText: {
    ...typography.small,
    color: colors.textMuted,
    textAlign: 'center',
    marginTop: spacing.sm,
  },
  deleteAllBtn: {
    marginTop: spacing.md,
    alignItems: 'center',
    paddingVertical: spacing.sm,
  },
  deleteAllText: {
    ...typography.caption,
    color: colors.error,
    fontWeight: '600',
  },
  bgRow: {
    flexDirection: 'row',
    alignItems: 'center',
    flexWrap: 'wrap',
    gap: spacing.sm,
    marginTop: spacing.md,
  },
  bgToggle: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.xs,
    borderRadius: radii.full,
    backgroundColor: colors.background,
    borderWidth: 1,
    borderColor: colors.border,
  },
  bgToggleActive: {
    backgroundColor: colors.success + '20',
    borderColor: colors.success,
  },
  bgToggleText: {
    ...typography.small,
    color: colors.textSecondary,
    fontWeight: '500',
  },
  bgMeta: {
    ...typography.small,
    color: colors.textMuted,
  },
  lossChart: {
    marginTop: spacing.md,
    marginBottom: spacing.sm,
  },
  lossChartTitle: {
    ...typography.small,
    color: colors.textSecondary,
    marginBottom: spacing.xs,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  lossChartInner: {
    height: 60,
    backgroundColor: colors.background,
    borderRadius: radii.sm,
    overflow: 'hidden',
    position: 'relative',
  },
  lossChartLabels: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: spacing.xs,
  },
  lossChartLabel: {
    ...typography.small,
    color: colors.textMuted,
  },
});
