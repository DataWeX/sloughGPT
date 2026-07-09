import React, {useEffect, useState, useRef} from 'react';
import {
  ScrollView,
  TextInput,
  Modal,
  RefreshControl,
  ActivityIndicator,
  Alert,
} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {YStack, XStack, Text} from 'tamagui';
import {useTrainingStore, type TrainPhase, type TrainingMethod} from '../stores/training-store';
import {useModelStore} from '../stores/model-store';
import {api} from '../services/api-client';
import {StatusBadge} from '../components/StatusBadge';
import {Icon} from '../components/Icon';

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
      <YStack height={80} backgroundColor="$background" borderRadius={4} alignItems="center" justifyContent="center">
        <Text fontSize={13} color="$color10">
          Loss curve will appear here
        </Text>
      </YStack>
    );
  }

  const maxLoss = Math.max(...data.map(d => d.value));
  const minLoss = Math.min(...data.map(d => d.value));
  const range = maxLoss - minLoss || 1;
  const W = 280;
  const H = 80;
  const pad = 4;

  return (
    <YStack marginTop={4}>
      <YStack backgroundColor="$background" borderRadius={4} overflow="hidden" style={{width: W, height: H}}>
        {data.map((point, i) => {
          const x = pad + (i / (data.length - 1)) * (W - pad * 2);
          const y = H - pad - ((point.value - minLoss) / range) * (H - pad * 2);
          const dotSize = i === data.length - 1 ? 6 : 3;
          const color = i === data.length - 1 ? '#7C52C4' : '#C0AAF4';
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
        <Text fontSize={11} color="$color10">{minLoss.toFixed(2)}</Text>
        <Text fontSize={11} color="$color10">{data.length} points</Text>
        <Text fontSize={11} color="$color10">{maxLoss.toFixed(2)}</Text>
      </XStack>
    </YStack>
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
    method,
    hfOpts,
    hfJobs,
    hfFinetunedPath,
    setConfig,
    setHfOpts,
    setMethod,
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
    <SafeAreaView style={{flex: 1, backgroundColor: '#F5F0FF'}} edges={['top']}>
      <ScrollView
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }>
        <YStack padding={16} gap={12}>
          <XStack alignItems="center" justifyContent="space-between">
            <Text fontSize={26} fontWeight="700" letterSpacing={-0.3} color="$color">Training</Text>
            <StatusBadge
              label={phaseInfo.text}
              variant={phaseInfo.variant as any}
            />
          </XStack>

          {error && (
            <XStack alignItems="center" justifyContent="space-between" backgroundColor="#FDE8E8" padding={12} borderRadius={8}>
              <Text fontSize={13} color="$red10" lineHeight={18} flex={1}>{error}</Text>
              <YStack onPress={clearError} pressStyle={{opacity: 0.7}}>
                <Icon name="x" size={16} color="#D44C56" />
              </YStack>
            </XStack>
          )}

          {!isTraining && !isDone && (
            <YStack backgroundColor="$background" borderRadius={12} borderWidth={1} borderColor="$borderColor" padding={16} gap={8}>
              <Text fontSize={16} fontWeight="600" color="$color" marginBottom={12}>Method</Text>
              <XStack gap={8} marginBottom={12}>
                <YStack
                  flex={1}
                  paddingVertical={8}
                  borderRadius={8}
                  backgroundColor={method === 'distill' ? '#7C52C4' : '$background'}
                  alignItems="center"
                  borderWidth={1}
                  borderColor={method === 'distill' ? '#7C52C4' : '$borderColor'}
                  onPress={() => setMethod('distill')}
                  pressStyle={{opacity: 0.7}}>
                  <Text fontSize={13} color={method === 'distill' ? '#FFFFFF' : '$color10'} fontWeight="500">
                    Distill
                  </Text>
                </YStack>
                <YStack
                  flex={1}
                  paddingVertical={8}
                  borderRadius={8}
                  backgroundColor={method === 'finetune' ? '#7C52C4' : '$background'}
                  alignItems="center"
                  borderWidth={1}
                  borderColor={method === 'finetune' ? '#7C52C4' : '$borderColor'}
                  onPress={() => setMethod('finetune')}
                  pressStyle={{opacity: 0.7}}>
                  <Text fontSize={13} color={method === 'finetune' ? '#FFFFFF' : '$color10'} fontWeight="500">
                    Fine-tune
                  </Text>
                </YStack>
              </XStack>
            </YStack>
          )}

          {!isTraining && !isDone && method === 'finetune' && (
            <YStack backgroundColor="$background" borderRadius={12} borderWidth={1} borderColor="$borderColor" padding={16} gap={8}>
              <Text fontSize={16} fontWeight="600" color="$color" marginBottom={12}>Fine-tune Settings</Text>
              <YStack marginBottom={12}>
                <Text fontSize={13} color="$color10" lineHeight={18} marginBottom={4}>Base Model</Text>
                <TextInput
                  value={hfOpts.model}
                  onChangeText={v => setHfOpts({model: v})}
                  placeholder="gpt2, Qwen/Qwen2.5-0.5B, ..."
                  placeholderTextColor="#9B95A8"
                  autoCapitalize="none"
                  autoCorrect={false}
                  style={{
                    fontSize: 15,
                    color: '#1A1625',
                    backgroundColor: '#F5F0FF',
                    borderRadius: 8,
                    paddingHorizontal: 12,
                    paddingVertical: 8,
                    borderWidth: 1,
                    borderColor: '#E0DCE8',
                    lineHeight: 22,
                  }}
                />
              </YStack>
              <YStack marginBottom={12}>
                <Text fontSize={13} color="$color10" lineHeight={18} marginBottom={4}>Dataset</Text>
                <YStack gap={4}>
                  {datasets.length === 0 ? (
                    <Text fontSize={13} color="$color10" lineHeight={18} textAlign="center" padding={16}>No datasets found. Import one first.</Text>
                  ) : (
                    datasets.map(ds => (
                      <XStack
                        key={ds.id}
                        alignItems="center"
                        justifyContent="space-between"
                        padding={12}
                        backgroundColor="$background"
                        borderRadius={8}
                        borderWidth={1}
                        borderColor={hfOpts.dataset === ds.id ? '#7C52C4' : '$borderColor'}
                        style={hfOpts.dataset === ds.id ? {backgroundColor: 'rgba(124, 82, 196, 0.1)'} : undefined}
                        onPress={() => setHfOpts({dataset: ds.id})}
                        pressStyle={{opacity: 0.7}}>
                        <YStack flex={1}>
                          <Text fontSize={15} color="$color" fontWeight="500" lineHeight={22}>{ds.name}</Text>
                          <Text fontSize={11} color="$color10" letterSpacing={0.2}>
                            {ds.file_count} files · {ds.total_chars.toLocaleString()} chars
                          </Text>
                        </YStack>
                        <YStack
                          paddingHorizontal={8}
                          paddingVertical={2}
                          borderRadius={4}
                          style={{backgroundColor: 'rgba(124, 82, 196, 0.15)'}}
                          onPress={() => fetchPreview(ds.id)}
                          pressStyle={{opacity: 0.7}}>
                          <Text fontSize={11} color="#7C52C4" fontWeight="500" letterSpacing={0.2}>Preview</Text>
                        </YStack>
                        {hfOpts.dataset === ds.id && <Icon name="check" size={18} color="#7C52C4" />}
                      </XStack>
                    ))
                  )}
                </YStack>
              </YStack>
              <YStack marginBottom={12}>
                <Text fontSize={13} color="$color10" lineHeight={18} marginBottom={4}>Epochs</Text>
                <XStack gap={4} flexWrap="wrap">
                  {[1, 2, 3, 5].map(v => (
                    <YStack
                      key={v}
                      paddingHorizontal={12}
                      paddingVertical={4}
                      borderRadius={9999}
                      backgroundColor={hfOpts.epochs === v ? '#7C52C4' : '$background'}
                      borderWidth={1}
                      borderColor={hfOpts.epochs === v ? '#7C52C4' : '$borderColor'}
                      onPress={() => setHfOpts({epochs: v})}
                      pressStyle={{opacity: 0.7}}>
                      <Text fontSize={11} color={hfOpts.epochs === v ? '#FFFFFF' : '$color10'} letterSpacing={0.2}>{v}</Text>
                    </YStack>
                  ))}
                </XStack>
              </YStack>
              <YStack marginBottom={12}>
                <Text fontSize={13} color="$color10" lineHeight={18} marginBottom={4}>Batch Size</Text>
                <XStack gap={4} flexWrap="wrap">
                  {[2, 4, 8, 16].map(v => (
                    <YStack
                      key={v}
                      paddingHorizontal={12}
                      paddingVertical={4}
                      borderRadius={9999}
                      backgroundColor={hfOpts.batch_size === v ? '#7C52C4' : '$background'}
                      borderWidth={1}
                      borderColor={hfOpts.batch_size === v ? '#7C52C4' : '$borderColor'}
                      onPress={() => setHfOpts({batch_size: v})}
                      pressStyle={{opacity: 0.7}}>
                      <Text fontSize={11} color={hfOpts.batch_size === v ? '#FFFFFF' : '$color10'} letterSpacing={0.2}>{v}</Text>
                    </YStack>
                  ))}
                </XStack>
              </YStack>
              <YStack marginBottom={12}>
                <Text fontSize={13} color="$color10" lineHeight={18} marginBottom={4}>Learning Rate</Text>
                <XStack gap={4} flexWrap="wrap">
                  {[1e-5, 2e-5, 5e-5, 1e-4].map(v => (
                    <YStack
                      key={v}
                      paddingHorizontal={12}
                      paddingVertical={4}
                      borderRadius={9999}
                      backgroundColor={Math.abs(hfOpts.learning_rate - v) < 1e-6 ? '#7C52C4' : '$background'}
                      borderWidth={1}
                      borderColor={Math.abs(hfOpts.learning_rate - v) < 1e-6 ? '#7C52C4' : '$borderColor'}
                      onPress={() => setHfOpts({learning_rate: v})}
                      pressStyle={{opacity: 0.7}}>
                      <Text fontSize={11} color={Math.abs(hfOpts.learning_rate - v) < 1e-6 ? '#FFFFFF' : '$color10'} letterSpacing={0.2}>{v.toExponential()}</Text>
                    </YStack>
                  ))}
                </XStack>
              </YStack>
              <YStack marginBottom={12}>
                <XStack alignItems="center" justifyContent="space-between">
                  <Text fontSize={13} color="$color10" lineHeight={18} marginBottom={4}>Use LoRA</Text>
                  <YStack
                    width={44}
                    height={24}
                    borderRadius={12}
                    backgroundColor={hfOpts.use_lora ? '#7C52C4' : '$borderColor'}
                    justifyContent="center"
                    paddingHorizontal={2}
                    onPress={() => setHfOpts({use_lora: !hfOpts.use_lora})}
                    pressStyle={{opacity: 0.7}}>
                    <YStack
                      width={20}
                      height={20}
                      borderRadius={10}
                      backgroundColor="#FFFFFF"
                      alignSelf={hfOpts.use_lora ? 'flex-end' : 'flex-start'}
                    />
                  </YStack>
                </XStack>
              </YStack>
              {hfOpts.use_lora && (
                <YStack marginBottom={12}>
                  <Text fontSize={13} color="$color10" lineHeight={18} marginBottom={4}>LoRA Rank</Text>
                  <XStack gap={4} flexWrap="wrap">
                    {[4, 8, 16, 32].map(v => (
                      <YStack
                        key={v}
                        paddingHorizontal={12}
                        paddingVertical={4}
                        borderRadius={9999}
                        backgroundColor={hfOpts.lora_rank === v ? '#7C52C4' : '$background'}
                        borderWidth={1}
                        borderColor={hfOpts.lora_rank === v ? '#7C52C4' : '$borderColor'}
                        onPress={() => setHfOpts({lora_rank: v})}
                        pressStyle={{opacity: 0.7}}>
                        <Text fontSize={11} color={hfOpts.lora_rank === v ? '#FFFFFF' : '$color10'} letterSpacing={0.2}>{v}</Text>
                      </YStack>
                    ))}
                  </XStack>
                </YStack>
              )}
            </YStack>
          )}

          {!isTraining && !isDone && method === 'distill' && (
            <YStack backgroundColor="$background" borderRadius={12} borderWidth={1} borderColor="$borderColor" padding={16} gap={8}>
              <Text fontSize={16} fontWeight="600" color="$color" marginBottom={12}>Training Data</Text>
              <XStack gap={8} marginBottom={12}>
                <YStack
                  flex={1}
                  paddingVertical={8}
                  borderRadius={8}
                  backgroundColor={inputMode === 'text' ? '#7C52C4' : '$background'}
                  alignItems="center"
                  borderWidth={1}
                  borderColor={inputMode === 'text' ? '#7C52C4' : '$borderColor'}
                  onPress={() => setInputMode('text')}
                  pressStyle={{opacity: 0.7}}>
                  <Text fontSize={13} color={inputMode === 'text' ? '#FFFFFF' : '$color10'} fontWeight="500">
                    Paste Text
                  </Text>
                </YStack>
                <YStack
                  flex={1}
                  paddingVertical={8}
                  borderRadius={8}
                  backgroundColor={inputMode === 'dataset' ? '#7C52C4' : '$background'}
                  alignItems="center"
                  borderWidth={1}
                  borderColor={inputMode === 'dataset' ? '#7C52C4' : '$borderColor'}
                  onPress={() => setInputMode('dataset')}
                  pressStyle={{opacity: 0.7}}>
                  <Text fontSize={13} color={inputMode === 'dataset' ? '#FFFFFF' : '$color10'} fontWeight="500">
                    Dataset
                  </Text>
                </YStack>
              </XStack>

              {inputMode === 'text' ? (
                <TextInput
                  value={sourceText}
                  onChangeText={setSourceText}
                  placeholder="Paste training text here (SRT, plain text, or lines)..."
                  placeholderTextColor="#9B95A8"
                  multiline
                  textAlignVertical="top"
                  style={{
                    fontSize: 15,
                    color: '#1A1625',
                    backgroundColor: '#F5F0FF',
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
                    <Text fontSize={13} color="$color10" lineHeight={18} textAlign="center" padding={16}>No datasets found</Text>
                  ) : (
                    datasets.map(ds => (
                      <XStack
                        key={ds.id}
                        alignItems="center"
                        justifyContent="space-between"
                        padding={12}
                        backgroundColor="$background"
                        borderRadius={8}
                        borderWidth={1}
                        borderColor={selectedDataset === ds.id ? '#7C52C4' : '$borderColor'}
                        style={selectedDataset === ds.id ? {backgroundColor: 'rgba(124, 82, 196, 0.1)'} : undefined}
                        onPress={() => setSelectedDataset(ds.id)}
                        pressStyle={{opacity: 0.7}}>
                        <YStack flex={1}>
                          <Text fontSize={15} color="$color" fontWeight="500" lineHeight={22}>{ds.name}</Text>
                          <Text fontSize={11} color="$color10" letterSpacing={0.2}>
                            {ds.file_count} files · {ds.total_chars.toLocaleString()} chars
                          </Text>
                        </YStack>
                        <YStack
                          paddingHorizontal={8}
                          paddingVertical={2}
                          borderRadius={4}
                          style={{backgroundColor: 'rgba(124, 82, 196, 0.15)'}}
                          onPress={() => fetchPreview(ds.id)}
                          pressStyle={{opacity: 0.7}}>
                          <Text fontSize={11} color="#7C52C4" fontWeight="500" letterSpacing={0.2}>Preview</Text>
                        </YStack>
                        {selectedDataset === ds.id && <Icon name="check" size={18} color="#7C52C4" />}
                      </XStack>
                    ))
                  )}
                </YStack>
              )}
            </YStack>
          )}

          {!isTraining && !isDone && method === 'distill' && (
            <YStack backgroundColor="$background" borderRadius={12} borderWidth={1} borderColor="$borderColor" padding={16} gap={8}>
              <Text fontSize={16} fontWeight="600" color="$color" marginBottom={12}>Hyperparameters</Text>
              <YStack marginBottom={12}>
                <Text fontSize={13} color="$color10" lineHeight={18} marginBottom={4}>Epochs</Text>
                <XStack gap={4} flexWrap="wrap">
                  {[3, 5, 10, 20, 50].map(v => (
                    <YStack
                      key={v}
                      paddingHorizontal={12}
                      paddingVertical={4}
                      borderRadius={9999}
                      backgroundColor={config.epochs === v ? '#7C52C4' : '$background'}
                      borderWidth={1}
                      borderColor={config.epochs === v ? '#7C52C4' : '$borderColor'}
                      onPress={() => setConfig({epochs: v})}
                      pressStyle={{opacity: 0.7}}>
                      <Text fontSize={11} color={config.epochs === v ? '#FFFFFF' : '$color10'} letterSpacing={0.2}>{v}</Text>
                    </YStack>
                  ))}
                </XStack>
              </YStack>
              <YStack marginBottom={12}>
                <Text fontSize={13} color="$color10" lineHeight={18} marginBottom={4}>Learning Rate</Text>
                <XStack gap={4} flexWrap="wrap">
                  {[0.0001, 0.001, 0.01].map(v => (
                    <YStack
                      key={v}
                      paddingHorizontal={12}
                      paddingVertical={4}
                      borderRadius={9999}
                      backgroundColor={config.learning_rate === v ? '#7C52C4' : '$background'}
                      borderWidth={1}
                      borderColor={config.learning_rate === v ? '#7C52C4' : '$borderColor'}
                      onPress={() => setConfig({learning_rate: v})}
                      pressStyle={{opacity: 0.7}}>
                      <Text fontSize={11} color={config.learning_rate === v ? '#FFFFFF' : '$color10'} letterSpacing={0.2}>{v}</Text>
                    </YStack>
                  ))}
                </XStack>
              </YStack>
              <YStack marginBottom={12}>
                <Text fontSize={13} color="$color10" lineHeight={18} marginBottom={4}>Soul</Text>
                <XStack gap={4} flexWrap="wrap">
                  {['assistant', 'creative', 'coder', 'teacher', 'analyst'].map(v => (
                    <YStack
                      key={v}
                      paddingHorizontal={12}
                      paddingVertical={4}
                      borderRadius={9999}
                      backgroundColor={config.soul_name === v ? '#7C52C4' : '$background'}
                      borderWidth={1}
                      borderColor={config.soul_name === v ? '#7C52C4' : '$borderColor'}
                      onPress={() => setConfig({soul_name: v})}
                      pressStyle={{opacity: 0.7}}>
                      <Text fontSize={11} color={config.soul_name === v ? '#FFFFFF' : '$color10'} letterSpacing={0.2}>{v}</Text>
                    </YStack>
                  ))}
                </XStack>
              </YStack>
            </YStack>
          )}

          {(isTraining || isDone || isFailed) && (
            <YStack backgroundColor="$background" borderRadius={12} borderWidth={1} borderColor="$borderColor" padding={16} gap={8}>
              <Text fontSize={16} fontWeight="600" color="$color" marginBottom={12}>Progress</Text>
              <XStack justifyContent="space-between" marginBottom={8}>
                <Text fontSize={15} color="$color" fontWeight="500" lineHeight={22}>Epoch {epoch}/{totalEpochs}</Text>
                <Text fontSize={15} color="$color" fontWeight="500" lineHeight={22}>{progress}%</Text>
              </XStack>
              <YStack height={8} backgroundColor="$borderColor" borderRadius={4} overflow="hidden" marginBottom={12}>
                <YStack
                  height="100%"
                  borderRadius={4}
                  backgroundColor={isDone ? '#2E9B7C' : isFailed ? '#D44C56' : '#7C52C4'}
                  width={`${progress}%`}
                />
              </YStack>
              <XStack gap={24} marginBottom={12}>
                <YStack>
                  <Text fontSize={11} color="$color10" letterSpacing={0.2}>Loss</Text>
                  <Text fontSize={16} fontWeight="600" color="$color">{loss !== null ? loss.toFixed(4) : '—'}</Text>
                </YStack>
                <YStack>
                  <Text fontSize={11} color="$color10" letterSpacing={0.2}>Steps</Text>
                  <Text fontSize={16} fontWeight="600" color="$color">{steps}</Text>
                </YStack>
              </XStack>
              <LossChart data={lossHistory} />
              {isTraining && (
                <XStack alignItems="center" gap={8} marginTop={12}>
                  <ActivityIndicator size="small" color="#7C52C4" />
                  <Text fontSize={13} color="$color10" lineHeight={18}>{phaseInfo.text}...</Text>
                </XStack>
              )}
            </YStack>
          )}

          {isDone && checkpoint && (
            <YStack backgroundColor="$background" borderRadius={12} borderWidth={1} borderColor="$borderColor" padding={16} gap={8}>
              <Text fontSize={16} fontWeight="600" color="$color" marginBottom={12}>Training Complete</Text>
              <StatusBadge label="Success" variant="success" />
              <Text fontSize={15} color="$color" lineHeight={22} marginTop={12}>Checkpoint: {checkpoint}</Text>
              <Text fontSize={11} color="$color10" letterSpacing={0.2} marginTop={4}>
                Final loss: {loss?.toFixed(4) || '—'} · {steps} steps
              </Text>
              <YStack
                marginTop={12}
                backgroundColor="#7C52C4"
                paddingVertical={8}
                paddingHorizontal={16}
                borderRadius={8}
                alignItems="center"
                onPress={() => handleLoadCheckpoint(checkpoint)}
                disabled={loadingCheckpoint === checkpoint}
                pressStyle={{opacity: 0.7}}>
                {loadingCheckpoint === checkpoint ? (
                  <ActivityIndicator size="small" color="#FFFFFF" />
                ) : (
                  <Text fontSize={13} color="#FFFFFF" fontWeight="600" lineHeight={18}>Load Model for Chat</Text>
                )}
              </YStack>
            </YStack>
          )}

          <YStack gap={8}>
            {isTraining ? (
              <YStack
                backgroundColor="#D44C56"
                paddingVertical={12}
                borderRadius={8}
                alignItems="center"
                onPress={stop}
                pressStyle={{opacity: 0.7}}>
                <Text fontSize={15} color="#FFFFFF" fontWeight="600" lineHeight={22}>Stop Training</Text>
              </YStack>
            ) : (
              <YStack
                backgroundColor="#7C52C4"
                paddingVertical={12}
                borderRadius={8}
                alignItems="center"
                opacity={running ? 0.5 : 1}
                onPress={handleStart}
                disabled={running}
                pressStyle={{opacity: 0.7}}>
                <Text fontSize={15} color="#FFFFFF" fontWeight="600" lineHeight={22}>
                  {isDone ? 'Train Again' : 'Start Training'}
                </Text>
              </YStack>
            )}
          </YStack>

          {isDone && hfFinetunedPath && (
            <YStack backgroundColor="$background" borderRadius={12} borderWidth={1} borderColor="$borderColor" padding={16} gap={8}>
              <Text fontSize={16} fontWeight="600" color="$color" marginBottom={12}>Fine-tune Complete</Text>
              <StatusBadge label="Success" variant="success" />
              <Text fontSize={15} color="$color" lineHeight={22} marginTop={12}>Model saved to: {hfFinetunedPath}</Text>
              <Text fontSize={11} color="$color10" letterSpacing={0.2} marginTop={4}>
                Loss: {loss?.toFixed(4) || '—'} · {steps} steps
              </Text>
              <YStack
                marginTop={12}
                backgroundColor="#7C52C4"
                paddingVertical={8}
                paddingHorizontal={16}
                borderRadius={8}
                alignItems="center"
                onPress={async () => {
                  try {
                    await modelStore.loadModel(hfFinetunedPath);
                    Alert.alert('Loaded', `Fine-tuned model loaded for chat`);
                  } catch (err: any) {
                    Alert.alert('Error', err.message || 'Failed to load model');
                  }
                }}
                pressStyle={{opacity: 0.7}}>
                <Text fontSize={13} color="#FFFFFF" fontWeight="600" lineHeight={18}>Load Model for Chat</Text>
              </YStack>
            </YStack>
          )}

          {hfJobs.length > 0 && (
            <YStack backgroundColor="$background" borderRadius={12} borderWidth={1} borderColor="$borderColor" padding={16} gap={8}>
              <Text fontSize={16} fontWeight="600" color="$color" marginBottom={12}>Job History</Text>
              {hfJobs.slice().reverse().map((job: any, i: number) => (
                <XStack key={job.job_id || job.id || i} alignItems="center" justifyContent="space-between" paddingVertical={8} borderBottomWidth={1} borderBottomColor="$borderColor">
                  <YStack flex={1} gap={4}>
                    <Text fontSize={15} color="$color" fontWeight="500" lineHeight={22}>
                      {job.model || 'Model'} · {job.dataset || 'dataset'}
                    </Text>
                    <Text fontSize={11} color="$color10" letterSpacing={0.2}>
                      Status: {job.status || job.phase || 'unknown'}{' '}
                      {job.loss != null ? `· Loss: ${Number(job.loss).toFixed(4)}` : ''}{' '}
                      {job.epoch != null ? `· Epoch ${job.epoch}` : ''}
                    </Text>
                  </YStack>
                  <StatusBadge
                    label={job.status === 'completed' ? 'Done' : job.status === 'failed' ? 'Failed' : job.status === 'running' ? 'Running' : job.phase || '—'}
                    variant={job.status === 'completed' ? 'success' : job.status === 'failed' ? 'error' : 'info'}
                  />
                </XStack>
              ))}
            </YStack>
          )}

          {checkpoints.length > 0 && (
            <YStack backgroundColor="$background" borderRadius={12} borderWidth={1} borderColor="$borderColor" padding={16} gap={8}>
              <Text fontSize={16} fontWeight="600" color="$color" marginBottom={12}>Checkpoints</Text>
              {checkpoints.map(cp => (
                <XStack key={cp.name} alignItems="center" justifyContent="space-between" paddingVertical={8} borderBottomWidth={1} borderBottomColor="$borderColor">
                  <YStack flex={1} gap={4}>
                    <Text fontSize={15} color="$color" fontWeight="500" lineHeight={22}>{cp.name}</Text>
                    <Text fontSize={11} color="$color10" letterSpacing={0.2}>
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
                          variant={cp.verdict === 'improved' ? 'success' : 'warning'}
                        />
                      )}
                    </XStack>
                  </YStack>
                  <XStack gap={4} alignItems="center">
                    <YStack
                      paddingHorizontal={12}
                      paddingVertical={4}
                      borderRadius={8}
                      style={{backgroundColor: 'rgba(124, 82, 196, 0.15)'}}
                      onPress={() => handleLoadCheckpoint(cp.name)}
                      disabled={loadingCheckpoint === cp.name}
                      pressStyle={{opacity: 0.7}}>
                      {loadingCheckpoint === cp.name ? (
                        <ActivityIndicator size="small" color="#7C52C4" />
                      ) : (
                        <Text fontSize={11} color="#7C52C4" fontWeight="600" letterSpacing={0.2}>Load</Text>
                      )}
                    </YStack>
                    <YStack
                      width={28}
                      height={28}
                      borderRadius={9999}
                      style={{backgroundColor: 'rgba(212, 76, 86, 0.15)'}}
                      alignItems="center"
                      justifyContent="center"
                      onPress={() => {
                        Alert.alert('Delete', `Delete ${cp.name}?`, [
                          {text: 'Cancel', style: 'cancel'},
                          {text: 'Delete', style: 'destructive', onPress: () => deleteCheckpoint(cp.name)},
                        ]);
                      }}
                      pressStyle={{opacity: 0.7}}>
                      <Icon name="x" size={16} color="#D44C56" />
                    </YStack>
                  </XStack>
                </XStack>
              ))}
            </YStack>
          )}
        </YStack>
      </ScrollView>

      <Modal visible={previewVisible} animationType="slide" transparent>
        <YStack flex={1} backgroundColor="rgba(0,0,0,0.4)" justifyContent="flex-end">
          <YStack backgroundColor="$background" borderTopLeftRadius={16} borderTopRightRadius={16} maxHeight="70%">
            <XStack alignItems="center" justifyContent="space-between" paddingHorizontal={20} paddingVertical={16} borderBottomWidth={1} borderBottomColor="$borderColor">
              <Text fontSize={16} fontWeight="600" color="$color">Dataset Preview</Text>
              <YStack onPress={() => setPreviewVisible(false)} pressStyle={{opacity: 0.7}}>
                <Icon name="x" size={24} color="#9B95A8" />
              </YStack>
            </XStack>
            <ScrollView style={{paddingHorizontal: 20, paddingVertical: 12}}>
              {previewData.map((line, i) => (
                <XStack key={i} gap={8} paddingVertical={4} borderBottomWidth={1} borderBottomColor="$borderColor">
                  <Text fontSize={11} color="$color10" letterSpacing={0.2} width={24}>{i + 1}</Text>
                  <Text fontSize={13} color="$color" lineHeight={18} flex={1} numberOfLines={3}>{line}</Text>
                </XStack>
              ))}
              {previewData.length === 0 && (
                <Text fontSize={13} color="$color10" lineHeight={18} textAlign="center" padding={24}>No preview available</Text>
              )}
            </ScrollView>
          </YStack>
        </YStack>
      </Modal>
    </SafeAreaView>
  );
}
