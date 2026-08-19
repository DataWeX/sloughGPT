import React, {useEffect, useState, useRef, useCallback} from 'react';
import {
  ScrollView,
  TextInput,
  Modal,
  RefreshControl,
  ActivityIndicator,
  Alert,
  Pressable,
} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {YStack, XStack, Text} from 'tamagui';
import {useColors} from '../theme/colors';
import {
  useTrainingStore,
  cleanupTraining,
  type TrainPhase,
} from '../stores/training-store';
import {useModelStore} from '../stores/model-store';
import {api} from '../services/api-client';
import {StatusBadge} from '../components/StatusBadge';
import {Icon} from '../components/Icon';
import {useHapticPress} from '../hooks/useHapticPress';
import {triggerHaptic} from '../services/haptics';

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

type ImportSource = 'url' | 'github' | 'huggingface' | 'csv';

const IMPORT_SOURCES: {key: ImportSource; label: string; placeholder: string}[] = [
  {key: 'url', label: 'URL', placeholder: 'https://example.com/data.txt'},
  {key: 'github', label: 'GitHub', placeholder: 'owner/repo or full URL'},
  {key: 'huggingface', label: 'HuggingFace', placeholder: 'dataset-id or org/dataset'},
  {key: 'csv', label: 'CSV', placeholder: 'https://example.com/data.csv'},
];

export function TrainingScreen() {
  const colors = useColors();
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
    method,
    hfOpts,
    hfJobs,
    hfFinetunedPath,
    finetunedModels,
    setConfig,
    setHfOpts,
    setMethod,
    start,
    stop,
    refresh,
    refreshFinetunedModels,
    loadCheckpoint,
    deleteCheckpoint,
    deleteJob,
    stopJob,
    loadFinetunedModel,
    deleteFinetunedModel,
    importDataset,
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
  const [showImportModal, setShowImportModal] = useState(false);
  const [importSource, setImportSource] = useState('');
  const [importName, setImportName] = useState('');
  const [importType, setImportType] = useState<ImportSource>('url');
  const [importing, setImporting] = useState(false);
  const [testPrompt, setTestPrompt] = useState('');
  const [testResult, setTestResult] = useState('');
  const [testLoading, setTestLoading] = useState(false);
  const [testModalVisible, setTestModalVisible] = useState(false);
  const prevPhaseRef = useRef(phase);
  const hapticPress = useHapticPress();

  useEffect(() => {
    if (prevPhaseRef.current !== 'COMPLETE' && phase === 'COMPLETE') {
      triggerHaptic('success');
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

  const prevErrorRef = useRef(error);
  useEffect(() => {
    if (prevErrorRef.current !== error && error) {
      triggerHaptic('error');
    }
    prevErrorRef.current = error;
  }, [error]);

  // Cleanup SSE + poll timers on unmount (BUG 1 fix)
  useEffect(() => {
    return () => {
      cleanupTraining();
    };
  }, []);

  useEffect(() => {
    refresh();
    refreshFinetunedModels();
  }, []);

  const fetchPreview = async (datasetId: string) => {
    try {
      const result = await api.get<{rows: string[]}>(`/datasets/${datasetId}/preview`);
      setPreviewData(result.rows || []);
      setPreviewVisible(true);
    } catch {}
  };

  const handleImport = async () => {
    if (!importSource.trim()) {
      Alert.alert('Import', 'Enter a source');
      return;
    }
    setImporting(true);
    try {
      await importDataset(importSource.trim(), importName.trim(), importType);
      triggerHaptic('success');
      Alert.alert('Imported', 'Dataset imported successfully');
      setShowImportModal(false);
      setImportSource('');
      setImportName('');
    } catch (err: any) {
      Alert.alert('Import Failed', err.message || 'Failed to import dataset');
    } finally {
      setImporting(false);
    }
  };

  const onRefresh = async () => {
    setRefreshing(true);
    await Promise.all([refresh(), refreshFinetunedModels()]);
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
    triggerHaptic('medium');
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

  const handleTestModel = async () => {
    if (!testPrompt.trim()) {
      return;
    }
    setTestLoading(true);
    setTestResult('');
    try {
      const result = await api.post<{text: string}>('/inference/generate', {
        prompt: testPrompt,
        max_new_tokens: 150,
      });
      setTestResult(result.text || 'No response');
    } catch (err: any) {
      setTestResult(`Error: ${err.message || 'Failed to generate'}`);
    } finally {
      setTestLoading(false);
    }
  };

  const isTraining =
    phase === 'TRAINING' ||
    phase === 'EVALUATING' ||
    phase === 'GENERATE_DATA' ||
    phase === 'DISTILL' ||
    phase === 'TRAIN' ||
    phase === 'EVALUATE' ||
    phase === 'DEPLOY';
  const isDone = phase === 'COMPLETE';
  const isFailed = phase === 'FAILED';
  const accent = colors.primary;
  const progress =
    totalEpochs > 0 ? Math.round((epoch / totalEpochs) * 100) : 0;
  const phaseInfo = PHASE_LABELS[phase] || PHASE_LABELS.idle;

  const LossChart = useCallback(
    ({data}: {data: {step: number; value: number}[]}) => {
      if (data.length < 2) {
        return (
          <YStack
            height={80}
            backgroundColor={colors.background}
            borderRadius={4}
            alignItems="center"
            justifyContent="center">
            <Text fontSize={13} color={colors.textSecondary}>
              Loss curve will appear here
            </Text>
          </YStack>
        );
      }

      // Use reduce instead of spread to avoid call-stack overflow on large arrays (BUG 4 fix)
      const maxLoss = data.reduce((max, d) => (d.value > max ? d.value : max), data[0].value);
      const minLoss = data.reduce((min, d) => (d.value < min ? d.value : min), data[0].value);
      const range = maxLoss - minLoss || 1;
      const W = 280;
      const H = 80;
      const pad = 4;

      return (
        <YStack marginTop={4}>
          <YStack
            backgroundColor={colors.background}
            borderRadius={4}
            overflow="hidden"
            style={{width: W, height: H}}>
            {data.map((point, i) => {
              const x = pad + (i / (data.length - 1)) * (W - pad * 2);
              const y =
                H -
                pad -
                ((point.value - minLoss) / range) * (H - pad * 2);
              const dotSize = i === data.length - 1 ? 6 : 3;
              const color =
                i === data.length - 1 ? colors.primary : '#C0AAF4';
              return (
                <YStack
                  key={i}
                  position="absolute"
                  style={{
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
          </YStack>
          <XStack justifyContent="space-between" marginTop={4}>
            <Text fontSize={11} color={colors.textSecondary}>
              {minLoss.toFixed(2)}
            </Text>
            <Text fontSize={11} color={colors.textSecondary}>
              {data.length} points
            </Text>
            <Text fontSize={11} color={colors.textSecondary}>
              {maxLoss.toFixed(2)}
            </Text>
          </XStack>
        </YStack>
      );
    },
    [colors],
  );

  // ── Section card wrapper ────────────────────────────────────────────────
  const Section = ({
    title,
    children,
  }: {
    title: string;
    children: React.ReactNode;
  }) => (
    <YStack
      backgroundColor={colors.background}
      borderRadius={12}
      borderWidth={0.5}
      borderColor={colors.border}
      padding={16}
      gap={8}>
      <Text
        fontSize={16}
        fontWeight="600"
        color={colors.text}
        marginBottom={12}>
        {title}
      </Text>
      {children}
    </YStack>
  );

  // ── Pill selector ──────────────────────────────────────────────────────
  const Pill = ({
    label,
    selected,
    onPress,
  }: {
    label: string;
    selected: boolean;
    onPress: () => void;
  }) => (
    <YStack
      paddingHorizontal={12}
      paddingVertical={4}
      borderRadius={999}
      backgroundColor={selected ? colors.primary : '$background'}
      borderWidth={0.5}
      borderColor={selected ? colors.primary : '$borderColor'}
      onPress={hapticPress('selection', onPress)}
      pressStyle={{opacity: 0.7}}>
      <Text
        fontSize={11}
        color={selected ? '#FFFFFF' : '$color10'}
        letterSpacing={0.2}>
        {label}
      </Text>
    </YStack>
  );

  return (
    <SafeAreaView style={{flex: 1}} edges={['top']}>
      <ScrollView
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }>
        <YStack padding={16} gap={12}>
          {/* ── Header ───────────────────────────────────────────────────── */}
          <XStack alignItems="center" justifyContent="space-between">
            <Text
              fontSize={26}
              fontWeight="700"
              letterSpacing={-0.3}
              color={colors.text}>
              Training
            </Text>
            <StatusBadge
              label={phaseInfo.text}
              variant={phaseInfo.variant as any}
            />
          </XStack>

          {/* ── Error banner ─────────────────────────────────────────────── */}
          {error && (
            <XStack
              alignItems="center"
              justifyContent="space-between"
              backgroundColor={colors.errorAlpha(0.08)}
              padding={12}
              borderRadius={10}>
              <Text
                fontSize={13}
                color={colors.error}
                lineHeight={18}
                flex={1}>
                {error}
              </Text>
              <Pressable
                onPress={hapticPress('light', clearError)}
                accessibilityLabel="Clear error">
                <Icon name="x" size={16} color={colors.error} />
              </Pressable>
            </XStack>
          )}

          {/* ── Method selector ──────────────────────────────────────────── */}
          {!isTraining && !isDone && (
            <Section title="Method">
              <XStack gap={8}>
                {(['distill', 'finetune'] as const).map(m => (
                  <YStack
                    key={m}
                    flex={1}
                    paddingVertical={8}
                    borderRadius={8}
                    backgroundColor={
                      method === m ? colors.primary : '$background'
                    }
                    alignItems="center"
                    borderWidth={0.5}
                    borderColor={
                      method === m ? accent : '$borderColor'
                    }
                    onPress={hapticPress('selection', () => setMethod(m))}
                    pressStyle={{opacity: 0.7}}>
                    <Text
                      fontSize={13}
                      color={method === m ? '#FFFFFF' : '$color10'}
                      fontWeight="500">
                      {m === 'distill' ? 'Distill' : 'Fine-tune'}
                    </Text>
                  </YStack>
                ))}
              </XStack>
            </Section>
          )}

          {/* ── Fine-tune settings ───────────────────────────────────────── */}
          {!isTraining && !isDone && method === 'finetune' && (
            <Section title="Fine-tune Settings">
              <YStack marginBottom={12}>
                <Text
                  fontSize={13}
                  color={colors.textSecondary}
                  lineHeight={18}
                  marginBottom={4}>
                  Base Model
                </Text>
                <TextInput
                  value={hfOpts.model}
                  onChangeText={v => setHfOpts({model: v})}
                  placeholder="gpt2, Qwen/Qwen2.5-0.5B, ..."
                  placeholderTextColor={colors.textMuted}
                  autoCapitalize="none"
                  autoCorrect={false}
                  style={{
                    fontSize: 15,
                    color: '#1A1625',
                    backgroundColor: colors.primaryAlpha(0.04),
                    borderRadius: 8,
                    paddingHorizontal: 12,
                    paddingVertical: 8,
                    borderWidth: 1,
                    borderColor: '#E4E0F2',
                    lineHeight: 22,
                  }}
                />
              </YStack>
              <YStack marginBottom={12}>
                <Text
                  fontSize={13}
                  color={colors.textSecondary}
                  lineHeight={18}
                  marginBottom={4}>
                  Dataset
                </Text>
                <YStack gap={4}>
                  {datasets.length === 0 ? (
                    <Text
                      fontSize={13}
                      color={colors.textSecondary}
                      lineHeight={18}
                      textAlign="center"
                      padding={16}>
                      No datasets found. Import one first.
                    </Text>
                  ) : (
                    datasets.map(ds => (
                      <XStack
                        key={ds.id}
                        alignItems="center"
                        justifyContent="space-between"
                        padding={12}
                        backgroundColor={colors.background}
                        borderRadius={8}
                        borderWidth={0.5}
                        borderColor={
                          hfOpts.dataset === ds.id
                            ? colors.primary
                            : '$borderColor'
                        }
                        style={
                          hfOpts.dataset === ds.id
                            ? {backgroundColor: colors.primaryAlpha(0.1)}
                            : undefined
                        }
                        onPress={hapticPress('selection', () =>
                          setHfOpts({dataset: ds.id}),
                        )}
                        pressStyle={{opacity: 0.7}}>
                        <YStack flex={1}>
                          <Text
                            fontSize={15}
                            color={colors.text}
                            fontWeight="500"
                            lineHeight={22}>
                            {ds.name}
                          </Text>
                          <Text
                            fontSize={11}
                            color={colors.textSecondary}
                            letterSpacing={0.2}>
                            {ds.file_count} files ·{' '}
                            {ds.total_chars.toLocaleString()} chars
                          </Text>
                        </YStack>
                        <YStack
                          paddingHorizontal={8}
                          paddingVertical={2}
                          borderRadius={4}
                          style={{
                            backgroundColor: colors.primaryAlpha(0.15),
                          }}
                          onPress={hapticPress('light', () =>
                            fetchPreview(ds.id),
                          )}
                          pressStyle={{opacity: 0.7}}>
                          <Text
                            fontSize={11}
                            color={colors.primary}
                            fontWeight="500"
                            letterSpacing={0.2}>
                            Preview
                          </Text>
                        </YStack>
                        {hfOpts.dataset === ds.id && (
                          <Icon
                            name="check"
                            size={18}
                            color={colors.primary}
                          />
                        )}
                      </XStack>
                    ))
                  )}
                </YStack>
              </YStack>
              <YStack marginBottom={12}>
                <Text
                  fontSize={13}
                  color={colors.textSecondary}
                  lineHeight={18}
                  marginBottom={4}>
                  Epochs
                </Text>
                <XStack gap={4} flexWrap="wrap">
                  {[1, 2, 3, 5].map(v => (
                    <Pill
                      key={v}
                      label={String(v)}
                      selected={hfOpts.epochs === v}
                      onPress={() => setHfOpts({epochs: v})}
                    />
                  ))}
                </XStack>
              </YStack>
              <YStack marginBottom={12}>
                <Text
                  fontSize={13}
                  color={colors.textSecondary}
                  lineHeight={18}
                  marginBottom={4}>
                  Batch Size
                </Text>
                <XStack gap={4} flexWrap="wrap">
                  {[2, 4, 8, 16].map(v => (
                    <Pill
                      key={v}
                      label={String(v)}
                      selected={hfOpts.batch_size === v}
                      onPress={() => setHfOpts({batch_size: v})}
                    />
                  ))}
                </XStack>
              </YStack>
              <YStack marginBottom={12}>
                <Text
                  fontSize={13}
                  color={colors.textSecondary}
                  lineHeight={18}
                  marginBottom={4}>
                  Learning Rate
                </Text>
                <XStack gap={4} flexWrap="wrap">
                  {[1e-5, 2e-5, 5e-5, 1e-4].map(v => (
                    <Pill
                      key={v}
                      label={v.toExponential()}
                      selected={
                        Math.abs(hfOpts.learning_rate - v) < 1e-6
                      }
                      onPress={() => setHfOpts({learning_rate: v})}
                    />
                  ))}
                </XStack>
              </YStack>
              <YStack marginBottom={12}>
                <XStack
                  alignItems="center"
                  justifyContent="space-between">
                  <Text
                    fontSize={13}
                    color={colors.textSecondary}
                    lineHeight={18}>
                    Use LoRA
                  </Text>
                  <YStack
                    width={44}
                    height={24}
                    borderRadius={12}
                    backgroundColor={
                      hfOpts.use_lora
                        ? colors.primary
                        : '$borderColor'
                    }
                    justifyContent="center"
                    paddingHorizontal={2}
                    onPress={hapticPress('selection', () =>
                      setHfOpts({use_lora: !hfOpts.use_lora}),
                    )}
                    pressStyle={{opacity: 0.7}}>
                    <YStack
                      width={20}
                      height={20}
                      borderRadius={10}
                      backgroundColor="white"
                      alignSelf={
                        hfOpts.use_lora ? 'flex-end' : 'flex-start'
                      }
                    />
                  </YStack>
                </XStack>
              </YStack>
              {hfOpts.use_lora && (
                <YStack marginBottom={12}>
                  <Text
                    fontSize={13}
                    color={colors.textSecondary}
                    lineHeight={18}
                    marginBottom={4}>
                    LoRA Rank
                  </Text>
                  <XStack gap={4} flexWrap="wrap">
                    {[4, 8, 16, 32].map(v => (
                      <Pill
                        key={v}
                        label={String(v)}
                        selected={hfOpts.lora_rank === v}
                        onPress={() => setHfOpts({lora_rank: v})}
                      />
                    ))}
                  </XStack>
                </YStack>
              )}
            </Section>
          )}

          {/* ── Distill: training data ───────────────────────────────────── */}
          {!isTraining && !isDone && method === 'distill' && (
            <Section title="Training Data">
              <XStack gap={8} marginBottom={12}>
                {(['text', 'dataset'] as const).map(mode => (
                  <YStack
                    key={mode}
                    flex={1}
                    paddingVertical={8}
                    borderRadius={8}
                    backgroundColor={
                      inputMode === mode
                        ? colors.primary
                        : '$background'
                    }
                    alignItems="center"
                    borderWidth={0.5}
                    borderColor={
                      inputMode === mode
                        ? colors.primary
                        : '$borderColor'
                    }
                    onPress={hapticPress('selection', () =>
                      setInputMode(mode),
                    )}
                    pressStyle={{opacity: 0.7}}>
                    <Text
                      fontSize={13}
                      color={
                        inputMode === mode ? '#FFFFFF' : '$color10'
                      }
                      fontWeight="500">
                      {mode === 'text' ? 'Paste Text' : 'Dataset'}
                    </Text>
                  </YStack>
                ))}
              </XStack>

              {inputMode === 'text' ? (
                <TextInput
                  value={sourceText}
                  onChangeText={setSourceText}
                  placeholder="Paste training text here (SRT, plain text, or lines)..."
                  placeholderTextColor={colors.textMuted}
                  multiline
                  textAlignVertical="top"
                  style={{
                    fontSize: 15,
                    color: '#1A1625',
                    backgroundColor: colors.primaryAlpha(0.04),
                    borderRadius: 8,
                    paddingHorizontal: 12,
                    paddingVertical: 12,
                    minHeight: 120,
                    lineHeight: 22,
                  }}
                />
              ) : (
                <YStack gap={4}>
                  {datasets.length === 0 ? (
                    <Text
                      fontSize={13}
                      color={colors.textSecondary}
                      lineHeight={18}
                      textAlign="center"
                      padding={16}>
                      No datasets found
                    </Text>
                  ) : (
                    datasets.map(ds => (
                      <XStack
                        key={ds.id}
                        alignItems="center"
                        justifyContent="space-between"
                        padding={12}
                        backgroundColor={colors.background}
                        borderRadius={8}
                        borderWidth={0.5}
                        borderColor={
                          selectedDataset === ds.id
                            ? colors.primary
                            : '$borderColor'
                        }
                        style={
                          selectedDataset === ds.id
                            ? {backgroundColor: colors.primaryAlpha(0.1)}
                            : undefined
                        }
                        onPress={hapticPress('selection', () =>
                          setSelectedDataset(ds.id),
                        )}
                        pressStyle={{opacity: 0.7}}>
                        <YStack flex={1}>
                          <Text
                            fontSize={15}
                            color={colors.text}
                            fontWeight="500"
                            lineHeight={22}>
                            {ds.name}
                          </Text>
                          <Text
                            fontSize={11}
                            color={colors.textSecondary}
                            letterSpacing={0.2}>
                            {ds.file_count} files ·{' '}
                            {ds.total_chars.toLocaleString()} chars
                          </Text>
                        </YStack>
                        <YStack
                          paddingHorizontal={8}
                          paddingVertical={2}
                          borderRadius={4}
                          style={{
                            backgroundColor: colors.primaryAlpha(0.15),
                          }}
                          onPress={hapticPress('light', () =>
                            fetchPreview(ds.id),
                          )}
                          pressStyle={{opacity: 0.7}}>
                          <Text
                            fontSize={11}
                            color={colors.primary}
                            fontWeight="500"
                            letterSpacing={0.2}>
                            Preview
                          </Text>
                        </YStack>
                        {selectedDataset === ds.id && (
                          <Icon
                            name="check"
                            size={18}
                            color={colors.primary}
                          />
                        )}
                      </XStack>
                    ))
                  )}
                  <YStack
                    paddingVertical={10}
                    borderRadius={8}
                    borderWidth={1}
                    borderStyle="dashed"
                    borderColor={colors.border}
                    alignItems="center"
                    onPress={hapticPress('light', () =>
                      setShowImportModal(true),
                    )}
                    pressStyle={{opacity: 0.7}}>
                    <Text
                      fontSize={13}
                      color={colors.primary}
                      fontWeight="500">
                      + Import Dataset
                    </Text>
                  </YStack>
                </YStack>
              )}
            </Section>
          )}

          {/* ── Distill: hyperparameters ─────────────────────────────────── */}
          {!isTraining && !isDone && method === 'distill' && (
            <Section title="Hyperparameters">
              <YStack marginBottom={12}>
                <Text
                  fontSize={13}
                  color={colors.textSecondary}
                  lineHeight={18}
                  marginBottom={4}>
                  Epochs
                </Text>
                <XStack gap={4} flexWrap="wrap">
                  {[3, 5, 10, 20, 50].map(v => (
                    <Pill
                      key={v}
                      label={String(v)}
                      selected={config.epochs === v}
                      onPress={() => setConfig({epochs: v})}
                    />
                  ))}
                </XStack>
              </YStack>
              <YStack marginBottom={12}>
                <Text
                  fontSize={13}
                  color={colors.textSecondary}
                  lineHeight={18}
                  marginBottom={4}>
                  Learning Rate
                </Text>
                <XStack gap={4} flexWrap="wrap">
                  {[0.0001, 0.001, 0.01].map(v => (
                    <Pill
                      key={v}
                      label={String(v)}
                      selected={config.learning_rate === v}
                      onPress={() => setConfig({learning_rate: v})}
                    />
                  ))}
                </XStack>
              </YStack>
              <YStack marginBottom={12}>
                <Text
                  fontSize={13}
                  color={colors.textSecondary}
                  lineHeight={18}
                  marginBottom={4}>
                  Soul
                </Text>
                <XStack gap={4} flexWrap="wrap">
                  {['assistant', 'creative', 'coder', 'teacher', 'analyst'].map(
                    v => (
                      <Pill
                        key={v}
                        label={v}
                        selected={config.soul_name === v}
                        onPress={() => setConfig({soul_name: v})}
                      />
                    ),
                  )}
                </XStack>
              </YStack>
            </Section>
          )}

          {/* ── Progress ─────────────────────────────────────────────────── */}
          {(isTraining || isDone || isFailed) && (
            <Section title="Progress">
              <XStack justifyContent="space-between" marginBottom={8}>
                <Text
                  fontSize={15}
                  color={colors.text}
                  fontWeight="500"
                  lineHeight={22}>
                  Epoch {epoch}/{totalEpochs}
                </Text>
                <Text
                  fontSize={15}
                  color={colors.text}
                  fontWeight="500"
                  lineHeight={22}>
                  {progress}%
                </Text>
              </XStack>
              <YStack
                height={8}
                backgroundColor={colors.border}
                borderRadius={4}
                overflow="hidden"
                marginBottom={12}>
                <YStack
                  height="100%"
                  borderRadius={4}
                  backgroundColor={
                    isDone
                      ? colors.successDark
                      : isFailed
                      ? colors.errorDark
                      : colors.primary
                  }
                  width={`${progress}%`}
                />
              </YStack>
              <XStack gap={24} marginBottom={12}>
                <YStack>
                  <Text
                    fontSize={11}
                    color={colors.textSecondary}
                    letterSpacing={0.2}>
                    Loss
                  </Text>
                  <Text
                    fontSize={16}
                    fontWeight="600"
                    color={colors.text}>
                    {loss !== null ? loss.toFixed(4) : '—'}
                  </Text>
                </YStack>
                <YStack>
                  <Text
                    fontSize={11}
                    color={colors.textSecondary}
                    letterSpacing={0.2}>
                    Steps
                  </Text>
                  <Text
                    fontSize={16}
                    fontWeight="600"
                    color={colors.text}>
                    {steps}
                  </Text>
                </YStack>
              </XStack>
              <LossChart data={lossHistory} />
              {isTraining && (
                <XStack alignItems="center" gap={8} marginTop={12}>
                  <ActivityIndicator size="small" color={colors.primary} />
                  <Text
                    fontSize={13}
                    color={colors.textSecondary}
                    lineHeight={18}>
                    {phaseInfo.text}...
                  </Text>
                </XStack>
              )}
            </Section>
          )}

          {/* ── Start / Stop button ──────────────────────────────────────── */}
          <YStack gap={8}>
            {isTraining ? (
              <YStack
                backgroundColor={colors.errorDark}
                paddingVertical={12}
                borderRadius={8}
                alignItems="center"
                onPress={hapticPress('medium', stop)}
                pressStyle={{opacity: 0.7}}>
                <Text
                  fontSize={15}
                  color={colors.white}
                  fontWeight="600"
                  lineHeight={22}>
                  Stop Training
                </Text>
              </YStack>
            ) : (
              <YStack
                backgroundColor={colors.primary}
                paddingVertical={12}
                borderRadius={8}
                alignItems="center"
                opacity={running ? 0.5 : 1}
                onPress={handleStart}
                disabled={running}
                pressStyle={{opacity: 0.7}}>
                <Text
                  fontSize={15}
                  color={colors.white}
                  fontWeight="600"
                  lineHeight={22}>
                  {isDone ? 'Train Again' : 'Start Training'}
                </Text>
              </YStack>
            )}
          </YStack>

          {/* ── Distill completion card ──────────────────────────────────── */}
          {isDone && checkpoint && (
            <Section title="Training Complete">
              <StatusBadge label="Success" variant="success" />
              <Text
                fontSize={15}
                color={colors.text}
                lineHeight={22}
                marginTop={12}>
                Checkpoint: {checkpoint}
              </Text>
              <Text
                fontSize={11}
                color={colors.textSecondary}
                letterSpacing={0.2}
                marginTop={4}>
                Final loss: {loss?.toFixed(4) || '—'} · {steps} steps
              </Text>
              <XStack gap={8} marginTop={12}>
                <YStack
                  flex={1}
                  backgroundColor={colors.primary}
                  paddingVertical={8}
                  paddingHorizontal={16}
                  borderRadius={8}
                  alignItems="center"
                  onPress={() => handleLoadCheckpoint(checkpoint)}
                  disabled={loadingCheckpoint === checkpoint}
                  pressStyle={{opacity: 0.7}}>
                  {loadingCheckpoint === checkpoint ? (
                    <ActivityIndicator size="small" color={colors.white} />
                  ) : (
                    <Text
                      fontSize={13}
                      color={colors.white}
                      fontWeight="600"
                      lineHeight={18}>
                      Load for Chat
                    </Text>
                  )}
                </YStack>
                <YStack
                  flex={1}
                  backgroundColor={colors.primaryAlpha(0.15)}
                  paddingVertical={8}
                  paddingHorizontal={16}
                  borderRadius={8}
                  alignItems="center"
                  onPress={hapticPress('light', () =>
                    setTestModalVisible(true),
                  )}
                  pressStyle={{opacity: 0.7}}>
                  <Text
                    fontSize={13}
                    color={colors.primary}
                    fontWeight="600"
                    lineHeight={18}>
                    Test Model
                  </Text>
                </YStack>
              </XStack>
            </Section>
          )}

          {/* ── Fine-tune completion card ────────────────────────────────── */}
          {isDone && hfFinetunedPath && (
            <Section title="Fine-tune Complete">
              <StatusBadge label="Success" variant="success" />
              <Text
                fontSize={15}
                color={colors.text}
                lineHeight={22}
                marginTop={12}>
                Model saved to: {hfFinetunedPath}
              </Text>
              <Text
                fontSize={11}
                color={colors.textSecondary}
                letterSpacing={0.2}
                marginTop={4}>
                Loss: {loss?.toFixed(4) || '—'} · {steps} steps
              </Text>
              <XStack gap={8} marginTop={12}>
                <YStack
                  flex={1}
                  backgroundColor={colors.primary}
                  paddingVertical={8}
                  paddingHorizontal={16}
                  borderRadius={8}
                  alignItems="center"
                  onPress={hapticPress('medium', async () => {
                    try {
                      await modelStore.loadModel(hfFinetunedPath);
                      Alert.alert('Loaded', 'Fine-tuned model loaded for chat');
                    } catch (err: any) {
                      Alert.alert(
                        'Error',
                        err.message || 'Failed to load model',
                      );
                    }
                  })}
                  pressStyle={{opacity: 0.7}}>
                  <Text
                    fontSize={13}
                    color={colors.white}
                    fontWeight="600"
                    lineHeight={18}>
                    Load for Chat
                  </Text>
                </YStack>
                <YStack
                  flex={1}
                  backgroundColor={colors.primaryAlpha(0.15)}
                  paddingVertical={8}
                  paddingHorizontal={16}
                  borderRadius={8}
                  alignItems="center"
                  onPress={hapticPress('light', () =>
                    setTestModalVisible(true),
                  )}
                  pressStyle={{opacity: 0.7}}>
                  <Text
                    fontSize={13}
                    color={colors.primary}
                    fontWeight="600"
                    lineHeight={18}>
                    Test Model
                  </Text>
                </YStack>
              </XStack>
            </Section>
          )}

          {/* ── Job History (with per-job stop + delete) ─────────────────── */}
          {hfJobs.length > 0 && (
            <Section title="Job History">
              {hfJobs.slice().reverse().map((job: any) => {
                const jobId = job.job_id || job.id;
                const isRunning =
                  job.status === 'running' || job.phase === 'TRAINING';
                return (
                  <XStack
                    key={jobId || Math.random()}
                    alignItems="center"
                    justifyContent="space-between"
                    paddingVertical={8}
                    borderBottomWidth={1}
                    borderBottomColor="$borderColor">
                    <YStack flex={1} gap={4}>
                      <Text
                        fontSize={15}
                        color={colors.text}
                        fontWeight="500"
                        lineHeight={22}>
                        {job.model || 'Model'} · {job.dataset || 'dataset'}
                      </Text>
                      <Text
                        fontSize={11}
                        color={colors.textSecondary}
                        letterSpacing={0.2}>
                        {job.status || job.phase || 'unknown'}{' '}
                        {job.loss != null
                          ? `· Loss: ${Number(job.loss).toFixed(4)}`
                          : ''}{' '}
                        {job.current_epoch != null
                          ? `· Epoch ${job.current_epoch}`
                          : ''}
                      </Text>
                    </YStack>
                    <XStack gap={4} alignItems="center">
                      {isRunning && jobId && (
                        <YStack
                          width={28}
                          height={28}
                          borderRadius={999}
                          style={{
                            backgroundColor: colors.primaryAlpha(0.15),
                          }}
                          alignItems="center"
                          justifyContent="center"
                          onPress={hapticPress('light', () =>
                            stopJob(jobId),
                          )}
                          pressStyle={{opacity: 0.7}}
                          accessible
                          accessibilityRole="button"
                          accessibilityLabel={`Stop job ${jobId}`}>
                          <Icon
                            name="x"
                            size={14}
                            color={colors.primary}
                          />
                        </YStack>
                      )}
                      {jobId && (
                        <YStack
                          width={28}
                          height={28}
                          borderRadius={999}
                          style={{
                            backgroundColor: 'rgba(212, 76, 86, 0.15)',
                          }}
                          alignItems="center"
                          justifyContent="center"
                          onPress={hapticPress('light', () => {
                            Alert.alert(
                              'Delete Job',
                              `Delete this training job?`,
                              [
                                {text: 'Cancel', style: 'cancel'},
                                {
                                  text: 'Delete',
                                  style: 'destructive',
                                  onPress: () => deleteJob(jobId),
                                },
                              ],
                            );
                          })}
                          pressStyle={{opacity: 0.7}}
                          accessible
                          accessibilityRole="button"
                          accessibilityLabel={`Delete job ${jobId}`}>
                          <Icon
                            name="trash-2"
                            size={14}
                            color={colors.errorDark}
                          />
                        </YStack>
                      )}
                    </XStack>
                  </XStack>
                );
              })}
            </Section>
          )}

          {/* ── Fine-tuned Models ────────────────────────────────────────── */}
          {finetunedModels.length > 0 && (
            <Section title="Fine-tuned Models">
              {finetunedModels.map(m => (
                <XStack
                  key={m.name}
                  alignItems="center"
                  justifyContent="space-between"
                  paddingVertical={8}
                  borderBottomWidth={1}
                  borderBottomColor="$borderColor">
                  <YStack flex={1} gap={4}>
                    <Text
                      fontSize={15}
                      color={colors.text}
                      fontWeight="500"
                      lineHeight={22}>
                      {m.name}
                    </Text>
                    <Text
                      fontSize={11}
                      color={colors.textSecondary}
                      letterSpacing={0.2}>
                      {m.model} · {m.dataset}{' '}
                      {m.size_mb ? `· ${m.size_mb.toFixed(1)} MB` : ''}{' '}
                      {m.final_loss != null
                        ? `· Loss: ${m.final_loss.toFixed(4)}`
                        : ''}
                    </Text>
                  </YStack>
                  <XStack gap={4} alignItems="center">
                    <YStack
                      paddingHorizontal={12}
                      paddingVertical={4}
                      borderRadius={8}
                      style={{
                        backgroundColor: colors.primaryAlpha(0.15),
                      }}
                      onPress={hapticPress('light', async () => {
                        try {
                          await loadFinetunedModel(m.name);
                          await modelStore.refresh();
                          Alert.alert(
                            'Loaded',
                            `${m.name} loaded for chat`,
                          );
                        } catch (err: any) {
                          Alert.alert(
                            'Error',
                            err.message || 'Failed to load model',
                          );
                        }
                      })}
                      pressStyle={{opacity: 0.7}}>
                      <Text
                        fontSize={11}
                        color={colors.primary}
                        fontWeight="600"
                        letterSpacing={0.2}>
                        Load
                      </Text>
                    </YStack>
                    <YStack
                      width={28}
                      height={28}
                      borderRadius={999}
                      style={{
                        backgroundColor: 'rgba(212, 76, 86, 0.15)',
                      }}
                      alignItems="center"
                      justifyContent="center"
                      onPress={hapticPress('light', () => {
                        Alert.alert(
                          'Delete',
                          `Delete ${m.name}?`,
                          [
                            {text: 'Cancel', style: 'cancel'},
                            {
                              text: 'Delete',
                              style: 'destructive',
                              onPress: () => deleteFinetunedModel(m.name),
                            },
                          ],
                        );
                      })}
                      pressStyle={{opacity: 0.7}}
                      accessible
                      accessibilityRole="button"
                      accessibilityLabel={`Delete fine-tuned model ${m.name}`}>
                      <Icon
                        name="x"
                        size={16}
                        color={colors.errorDark}
                      />
                    </YStack>
                  </XStack>
                </XStack>
              ))}
            </Section>
          )}

          {/* ── Checkpoints ──────────────────────────────────────────────── */}
          {checkpoints.length > 0 && (
            <Section title="Checkpoints">
              {checkpoints.map(cp => (
                <XStack
                  key={cp.name}
                  alignItems="center"
                  justifyContent="space-between"
                  paddingVertical={8}
                  borderBottomWidth={1}
                  borderBottomColor="$borderColor">
                  <YStack flex={1} gap={4}>
                    <Text
                      fontSize={15}
                      color={colors.text}
                      fontWeight="500"
                      lineHeight={22}>
                      {cp.name}
                    </Text>
                    <Text
                      fontSize={11}
                      color={colors.textSecondary}
                      letterSpacing={0.2}>
                      {cp.loss !== null ? `Loss: ${cp.loss.toFixed(3)}` : ''}{' '}
                      {cp.steps > 0 ? `· ${cp.steps} steps` : ''}{' '}
                      {cp.size_mb ? `· ${cp.size_mb} MB` : ''}
                    </Text>
                    <XStack gap={4}>
                      {cp.soul && cp.soul !== 'unknown' && (
                        <StatusBadge label={cp.soul} variant="info" />
                      )}
                      {cp.verdict && (
                        <StatusBadge
                          label={cp.verdict}
                          variant={
                            cp.verdict === 'improved'
                              ? 'success'
                              : 'warning'
                          }
                        />
                      )}
                    </XStack>
                  </YStack>
                  <XStack gap={4} alignItems="center">
                    <YStack
                      paddingHorizontal={12}
                      paddingVertical={4}
                      borderRadius={8}
                      style={{
                        backgroundColor: colors.primaryAlpha(0.15),
                      }}
                      onPress={() => handleLoadCheckpoint(cp.name)}
                      disabled={loadingCheckpoint === cp.name}
                      pressStyle={{opacity: 0.7}}>
                      {loadingCheckpoint === cp.name ? (
                        <ActivityIndicator
                          size="small"
                          color={colors.primary}
                        />
                      ) : (
                        <Text
                          fontSize={11}
                          color={colors.primary}
                          fontWeight="600"
                          letterSpacing={0.2}>
                          Load
                        </Text>
                      )}
                    </YStack>
                    <YStack
                      width={28}
                      height={28}
                      borderRadius={999}
                      style={{
                        backgroundColor: 'rgba(212, 76, 86, 0.15)',
                      }}
                      alignItems="center"
                      justifyContent="center"
                      onPress={hapticPress('light', () => {
                        Alert.alert(
                          'Delete',
                          `Delete ${cp.name}?`,
                          [
                            {text: 'Cancel', style: 'cancel'},
                            {
                              text: 'Delete',
                              style: 'destructive',
                              onPress: () => deleteCheckpoint(cp.name),
                            },
                          ],
                        );
                      })}
                      pressStyle={{opacity: 0.7}}
                      accessible
                      accessibilityRole="button"
                      accessibilityLabel={`Delete checkpoint ${cp.name}`}>
                      <Icon
                        name="x"
                        size={16}
                        color={colors.errorDark}
                      />
                    </YStack>
                  </XStack>
                </XStack>
              ))}
            </Section>
          )}
        </YStack>
      </ScrollView>

      {/* ── Dataset Preview Modal ──────────────────────────────────────── */}
      <Modal visible={previewVisible} animationType="slide" transparent>
        <YStack
          flex={1}
          backgroundColor="rgba(0,0,0,0.4)"
          justifyContent="flex-end">
          <YStack
            backgroundColor={colors.background}
            borderTopLeftRadius={24}
            borderTopRightRadius={24}
            maxHeight="70%">
            <XStack
              alignItems="center"
              justifyContent="space-between"
              paddingHorizontal={20}
              paddingVertical={16}
              borderBottomWidth={1}
              borderBottomColor="$borderColor">
              <Text
                fontSize={16}
                fontWeight="600"
                color={colors.text}>
                Dataset Preview
              </Text>
              <Pressable
                onPress={hapticPress('light', () => setPreviewVisible(false))}
                accessibilityLabel="Close preview">
                <YStack
                  width={28}
                  height={28}
                  borderRadius={9}
                  alignItems="center"
                  justifyContent="center">
                  <Icon name="x" size={16} color={colors.textSecondary} />
                </YStack>
              </Pressable>
            </XStack>
            <ScrollView
              style={{paddingHorizontal: 20, paddingVertical: 12}}>
              {previewData.map((line, i) => (
                <XStack
                  key={i}
                  gap={8}
                  paddingVertical={4}
                  borderBottomWidth={1}
                  borderBottomColor="$borderColor">
                  <Text
                    fontSize={11}
                    color={colors.textSecondary}
                    letterSpacing={0.2}
                    width={24}>
                    {i + 1}
                  </Text>
                  <Text
                    fontSize={13}
                    color={colors.text}
                    lineHeight={18}
                    flex={1}
                    numberOfLines={3}>
                    {line}
                  </Text>
                </XStack>
              ))}
              {previewData.length === 0 && (
                <Text
                  fontSize={13}
                  color={colors.textSecondary}
                  lineHeight={18}
                  textAlign="center"
                  padding={24}>
                  No preview available
                </Text>
              )}
            </ScrollView>
          </YStack>
        </YStack>
      </Modal>

      {/* ── Import Dataset Modal (multi-source) ────────────────────────── */}
      <Modal visible={showImportModal} animationType="slide" transparent>
        <YStack
          flex={1}
          backgroundColor="rgba(0,0,0,0.4)"
          justifyContent="flex-end">
          <YStack
            backgroundColor={colors.background}
            borderTopLeftRadius={24}
            borderTopRightRadius={24}>
            <XStack
              alignItems="center"
              justifyContent="space-between"
              paddingHorizontal={20}
              paddingVertical={16}
              borderBottomWidth={1}
              borderBottomColor="$borderColor">
              <Text
                fontSize={16}
                fontWeight="600"
                color={colors.text}>
                Import Dataset
              </Text>
              <Pressable
                onPress={hapticPress('light', () => setShowImportModal(false))}
                accessibilityLabel="Close import">
                <YStack
                  width={28}
                  height={28}
                  borderRadius={9}
                  alignItems="center"
                  justifyContent="center">
                  <Icon name="x" size={16} color={colors.textSecondary} />
                </YStack>
              </Pressable>
            </XStack>
            <YStack padding={20} gap={16}>
              <YStack gap={8}>
                <Text
                  fontSize={13}
                  color={colors.textSecondary}
                  fontWeight="500">
                  Source
                </Text>
                <XStack gap={4} flexWrap="wrap">
                  {IMPORT_SOURCES.map(src => (
                    <Pill
                      key={src.key}
                      label={src.label}
                      selected={importType === src.key}
                      onPress={() => {
                        setImportType(src.key);
                        setImportSource('');
                      }}
                    />
                  ))}
                </XStack>
              </YStack>
              <YStack gap={4}>
                <Text fontSize={13} color={colors.textSecondary}>
                  {importType === 'github'
                    ? 'Repository'
                    : importType === 'huggingface'
                    ? 'Dataset ID'
                    : 'URL or Path'}
                </Text>
                <TextInput
                  value={importSource}
                  onChangeText={setImportSource}
                  placeholder={
                    IMPORT_SOURCES.find(s => s.key === importType)
                      ?.placeholder
                  }
                  placeholderTextColor={colors.textMuted}
                  autoCapitalize="none"
                  autoCorrect={false}
                  style={{
                    fontSize: 15,
                    color: '#1A1625',
                    backgroundColor: colors.primaryAlpha(0.04),
                    borderRadius: 8,
                    paddingHorizontal: 12,
                    paddingVertical: 10,
                  }}
                />
              </YStack>
              <YStack gap={4}>
                <Text fontSize={13} color={colors.textSecondary}>
                  Name (optional)
                </Text>
                <TextInput
                  value={importName}
                  onChangeText={setImportName}
                  placeholder="my-dataset"
                  placeholderTextColor={colors.textMuted}
                  autoCapitalize="none"
                  autoCorrect={false}
                  style={{
                    fontSize: 15,
                    color: '#1A1625',
                    backgroundColor: colors.primaryAlpha(0.04),
                    borderRadius: 8,
                    paddingHorizontal: 12,
                    paddingVertical: 10,
                  }}
                />
              </YStack>
              <YStack
                paddingVertical={12}
                borderRadius={10}
                alignItems="center"
                backgroundColor={
                  importing || !importSource.trim()
                    ? 'rgba(124, 82, 196, 0.3)'
                    : colors.primary
                }
                onPress={hapticPress('light', handleImport)}
                disabled={importing || !importSource.trim()}
                pressStyle={{opacity: 0.7}}>
                {importing ? (
                  <ActivityIndicator color={colors.white} />
                ) : (
                  <Text
                    fontSize={14}
                    fontWeight="600"
                    color={colors.white}>
                    Import
                  </Text>
                )}
              </YStack>
            </YStack>
          </YStack>
        </YStack>
      </Modal>

      {/* ── Test Model Modal ───────────────────────────────────────────── */}
      <Modal visible={testModalVisible} animationType="slide" transparent>
        <YStack
          flex={1}
          backgroundColor="rgba(0,0,0,0.4)"
          justifyContent="flex-end">
          <YStack
            backgroundColor={colors.background}
            borderTopLeftRadius={24}
            borderTopRightRadius={24}
            maxHeight="80%">
            <XStack
              alignItems="center"
              justifyContent="space-between"
              paddingHorizontal={20}
              paddingVertical={16}
              borderBottomWidth={1}
              borderBottomColor="$borderColor">
              <Text
                fontSize={16}
                fontWeight="600"
                color={colors.text}>
                Test Model
              </Text>
              <Pressable
                onPress={hapticPress('light', () => {
                  setTestModalVisible(false);
                  setTestResult('');
                  setTestPrompt('');
                })}
                accessibilityLabel="Close test">
                <YStack
                  width={28}
                  height={28}
                  borderRadius={9}
                  alignItems="center"
                  justifyContent="center">
                  <Icon name="x" size={16} color={colors.textSecondary} />
                </YStack>
              </Pressable>
            </XStack>
            <YStack padding={20} gap={12}>
              <YStack gap={4}>
                <Text fontSize={13} color={colors.textSecondary}>
                  Prompt
                </Text>
                <TextInput
                  value={testPrompt}
                  onChangeText={setTestPrompt}
                  placeholder="Type a prompt to test the trained model..."
                  placeholderTextColor={colors.textMuted}
                  autoCapitalize="none"
                  autoCorrect={false}
                  style={{
                    fontSize: 15,
                    color: '#1A1625',
                    backgroundColor: colors.primaryAlpha(0.04),
                    borderRadius: 8,
                    paddingHorizontal: 12,
                    paddingVertical: 10,
                    minHeight: 60,
                  }}
                />
              </YStack>
              <YStack
                paddingVertical={12}
                borderRadius={10}
                alignItems="center"
                backgroundColor={
                  testLoading || !testPrompt.trim()
                    ? 'rgba(124, 82, 196, 0.3)'
                    : colors.primary
                }
                onPress={hapticPress('light', handleTestModel)}
                disabled={testLoading || !testPrompt.trim()}
                pressStyle={{opacity: 0.7}}>
                {testLoading ? (
                  <ActivityIndicator color={colors.white} />
                ) : (
                  <Text
                    fontSize={14}
                    fontWeight="600"
                    color={colors.white}>
                    Generate
                  </Text>
                )}
              </YStack>
              {testResult ? (
                <YStack gap={4}>
                  <Text fontSize={13} color={colors.textSecondary}>
                    Response
                  </Text>
                  <YStack
                    backgroundColor={colors.primaryAlpha(0.04)}
                    borderRadius={8}
                    padding={12}>
                    <Text
                      fontSize={14}
                      color={colors.text}
                      lineHeight={20}>
                      {testResult}
                    </Text>
                  </YStack>
                </YStack>
              ) : null}
            </YStack>
          </YStack>
        </YStack>
      </Modal>
    </SafeAreaView>
  );
}
