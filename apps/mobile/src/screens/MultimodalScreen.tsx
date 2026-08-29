import React, {useEffect, useState, useCallback, useRef} from 'react';
import {FlatList, Pressable, RefreshControl, TextInput as RNTextInput} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {YStack, XStack, Text} from 'tamagui';
import {useColors} from '../theme/colors';
import {api} from '../services/api-client';
import {Icon} from '../components/Icon';
import {StatusBadge} from '../components/StatusBadge';
import {triggerHaptic} from '../services/haptics';
import {toast} from '../services/toast';
import {startRecording, transcribeAudio} from '../services/voice-input';

type Tab = 'status' | 'vision' | 'audio' | 'generate';

interface MultimodalStatus {
  engine: {
    trained: boolean;
    vocab_size: number;
    learning_count: number;
    unique_captions: number;
  };
  vision: {available: boolean};
  audio: {available: boolean; tts_calls: number};
  dpo: {running: boolean; status: string | null};
  video: {training: boolean; progress: number};
}

export function MultimodalScreen() {
  const colors = useColors();
  const [tab, setTab] = useState<Tab>('status');
  const [status, setStatus] = useState<MultimodalStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  // Vision
  const [visionLabel, setVisionLabel] = useState('');
  const [trainingVision, setTrainingVision] = useState(false);

  // Audio / Transcribe
  const [transcribeResult, setTranscribeResult] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const recordingRef = useRef<{stop: () => Promise<{uri: string; duration: number} | null>} | null>(null);

  // Image generation
  const [genPrompt, setGenPrompt] = useState('');
  const [generating, setGenerating] = useState(false);

  // DPO
  const [dpoRunning, setDpoRunning] = useState(false);

  const fetchStatus = useCallback(async () => {
    try {
      const data = await api.get<MultimodalStatus>('/multimodal/status');
      setStatus(data);
    } catch {
      setStatus(null);
    }
  }, []);

  useEffect(() => {
    fetchStatus().finally(() => setLoading(false));
  }, [fetchStatus]);

  const onRefresh = async () => {
    setRefreshing(true);
    await fetchStatus();
    setRefreshing(false);
  };

  const handleTrainVision = async () => {
    try {
      setTrainingVision(true);
      triggerHaptic('light');
      await api.post('/multimodal/train', {label: visionLabel.trim() || undefined});
      triggerHaptic('success');
      toast.success('Vision training started');
      setVisionLabel('');
      await fetchStatus();
    } catch {
      toast.error('Training failed');
    } finally {
      setTrainingVision(false);
    }
  };

  const handleTranscribe = async () => {
    try {
      triggerHaptic('light');
      if (isRecording && recordingRef.current) {
        // Stop recording and transcribe
        setIsRecording(false);
        const recording = await recordingRef.current.stop();
        recordingRef.current = null;
        if (recording) {
          const text = await transcribeAudio(recording.uri);
          setTranscribeResult(text || 'No speech detected');
          if (text) {
            toast.success('Transcription complete');
          } else {
            toast.warn('No speech detected in recording');
          }
        }
      } else {
        // Start recording
        const {stop} = await startRecording();
        recordingRef.current = {stop};
        setIsRecording(true);
        toast.success('Recording started — tap again to stop');
      }
    } catch {
      setIsRecording(false);
      recordingRef.current = null;
      toast.error('Transcription failed');
    }
  };

  const handleGenerateImage = async () => {
    if (!genPrompt.trim()) return;
    try {
      setGenerating(true);
      triggerHaptic('light');
      await api.post('/multimodal/generate-image', {prompt: genPrompt.trim()});
      triggerHaptic('success');
      toast.success('Image generation started');
      setGenPrompt('');
    } catch {
      toast.error('Generation failed');
    } finally {
      setGenerating(false);
    }
  };

  const handleTriggerDPO = async () => {
    try {
      setDpoRunning(true);
      triggerHaptic('light');
      await api.post('/multimodal/dpo');
      triggerHaptic('success');
      toast.success('DPO training triggered');
      await fetchStatus();
    } catch {
      toast.error('DPO failed');
    } finally {
      setDpoRunning(false);
    }
  };

  const TABS: {key: Tab; label: string; icon: string}[] = [
    {key: 'status', label: 'Status', icon: 'info'},
    {key: 'vision', label: 'Vision', icon: 'image'},
    {key: 'audio', label: 'Audio', icon: 'music'},
    {key: 'generate', label: 'Generate', icon: 'plus'},
  ];

  const renderTabContent = () => {
    switch (tab) {
      case 'status':
        return (
          <YStack gap={10}>
            {/* Engine Status */}
            <YStack padding={14} borderRadius={10} backgroundColor={colors.white} borderWidth={0.5} borderColor={colors.border} gap={8}>
              <XStack justifyContent="space-between" alignItems="center">
                <Text fontSize={15} fontWeight="600" color={colors.text}>Vision Engine</Text>
                <StatusBadge label={status?.engine.trained ? 'Trained' : 'Untrained'} variant={status?.engine.trained ? 'success' : 'default'} />
              </XStack>
              <XStack gap={16}>
                <YStack gap={2}>
                  <Text fontSize={11} color={colors.textMuted}>Vocab</Text>
                  <Text fontSize={13} fontWeight="500" color={colors.text}>{status?.engine.vocab_size ?? 0}</Text>
                </YStack>
                <YStack gap={2}>
                  <Text fontSize={11} color={colors.textMuted}>Learned</Text>
                  <Text fontSize={13} fontWeight="500" color={colors.text}>{status?.engine.learning_count ?? 0}</Text>
                </YStack>
                <YStack gap={2}>
                  <Text fontSize={11} color={colors.textMuted}>Captions</Text>
                  <Text fontSize={13} fontWeight="500" color={colors.text}>{status?.engine.unique_captions ?? 0}</Text>
                </YStack>
              </XStack>
            </YStack>

            {/* DPO */}
            <YStack padding={14} borderRadius={10} backgroundColor={colors.white} borderWidth={0.5} borderColor={colors.border} gap={8}>
              <XStack justifyContent="space-between" alignItems="center">
                <Text fontSize={15} fontWeight="600" color={colors.text}>DPO Training</Text>
                <StatusBadge label={status?.dpo.running ? 'Running' : 'Idle'} variant={status?.dpo.running ? 'warning' : 'default'} />
              </XStack>
              <Pressable onPress={handleTriggerDPO} disabled={dpoRunning || status?.dpo.running}>
                <XStack padding={10} borderRadius={8} backgroundColor={!dpoRunning && !status?.dpo.running ? colors.primary : colors.border} alignItems="center" justifyContent="center" gap={6}>
                  <Icon name="zap" size={16} color="white" />
                  <Text fontSize={13} fontWeight="600" color="white">{dpoRunning ? 'Starting...' : 'Run DPO'}</Text>
                </XStack>
              </Pressable>
            </YStack>

            {/* Video */}
            <YStack padding={14} borderRadius={10} backgroundColor={colors.white} borderWidth={0.5} borderColor={colors.border} gap={8}>
              <XStack justifyContent="space-between" alignItems="center">
                <Text fontSize={15} fontWeight="600" color={colors.text}>Video Training</Text>
                <StatusBadge label={status?.video.training ? 'Training' : 'Idle'} variant={status?.video.training ? 'warning' : 'default'} />
              </XStack>
              {status?.video.training && (
                <YStack gap={4}>
                  <XStack justifyContent="space-between">
                    <Text fontSize={12} color={colors.textMuted}>Progress</Text>
                    <Text fontSize={12} fontWeight="500" color={colors.text}>{Math.round((status.video.progress || 0) * 100)}%</Text>
                  </XStack>
                  <YStack height={6} borderRadius={3} backgroundColor={colors.border}>
                    <YStack height={6} borderRadius={3} backgroundColor={colors.primary} width={`${Math.round((status.video.progress || 0) * 100)}%`} />
                  </YStack>
                </YStack>
              )}
            </YStack>
          </YStack>
        );

      case 'vision':
        return (
          <YStack gap={10}>
            <YStack padding={14} borderRadius={10} backgroundColor={colors.white} borderWidth={0.5} borderColor={colors.border} gap={8}>
              <Text fontSize={15} fontWeight="600" color={colors.text}>Train on Image</Text>
              <Text fontSize={12} color={colors.textMuted}>Upload an image to train the vision model. The model learns to describe images.</Text>
              <RNTextInput
                value={visionLabel}
                onChangeText={setVisionLabel}
                placeholder="Optional label"
                placeholderTextColor={colors.textMuted}
                style={{borderWidth: 1, borderColor: colors.border, borderRadius: 8, padding: 8, fontSize: 13, color: colors.text, backgroundColor: colors.background}}
              />
              <Pressable onPress={handleTrainVision} disabled={trainingVision}>
                <XStack padding={10} borderRadius={8} backgroundColor={!trainingVision ? colors.primary : colors.border} alignItems="center" justifyContent="center" gap={6}>
                  <Icon name={trainingVision ? 'refresh-cw' : 'upload'} size={16} color="white" />
                  <Text fontSize={13} fontWeight="600" color="white">{trainingVision ? 'Training...' : 'Train'}</Text>
                </XStack>
              </Pressable>
            </YStack>
          </YStack>
        );

      case 'audio':
        return (
          <YStack gap={10}>
            <YStack padding={14} borderRadius={10} backgroundColor={colors.white} borderWidth={0.5} borderColor={colors.border} gap={8}>
              <Text fontSize={15} fontWeight="600" color={colors.text}>Transcribe Audio</Text>
              <Text fontSize={12} color={colors.textMuted}>{isRecording ? 'Recording... tap to stop and transcribe.' : 'Tap to record audio. Tap again to stop and transcribe.'}</Text>
              <Pressable onPress={handleTranscribe}>
                <XStack padding={10} borderRadius={8} backgroundColor={isRecording ? colors.error : colors.primary} alignItems="center" justifyContent="center" gap={6}>
                  <Icon name={isRecording ? 'square' : 'mic'} size={16} color="white" />
                  <Text fontSize={13} fontWeight="600" color="white">{isRecording ? 'Stop & Transcribe' : 'Record & Transcribe'}</Text>
                </XStack>
              </Pressable>
              {transcribeResult ? (
                <YStack padding={10} borderRadius={6} backgroundColor={colors.background}>
                  <Text fontSize={13} color={colors.text}>{transcribeResult}</Text>
                </YStack>
              ) : null}
            </YStack>
          </YStack>
        );

      case 'generate':
        return (
          <YStack gap={10}>
            <YStack padding={14} borderRadius={10} backgroundColor={colors.white} borderWidth={0.5} borderColor={colors.border} gap={8}>
              <Text fontSize={15} fontWeight="600" color={colors.text}>Generate Image</Text>
              <RNTextInput
                value={genPrompt}
                onChangeText={setGenPrompt}
                placeholder="Describe the image to generate..."
                placeholderTextColor={colors.textMuted}
                multiline
                numberOfLines={3}
                style={{borderWidth: 1, borderColor: colors.border, borderRadius: 8, padding: 8, fontSize: 13, color: colors.text, backgroundColor: colors.background, minHeight: 72, textAlignVertical: 'top'}}
              />
              <Pressable onPress={handleGenerateImage} disabled={!genPrompt.trim() || generating}>
                <XStack padding={10} borderRadius={8} backgroundColor={genPrompt.trim() && !generating ? colors.primary : colors.border} alignItems="center" justifyContent="center" gap={6}>
                  <Icon name={generating ? 'refresh-cw' : 'image'} size={16} color="white" />
                  <Text fontSize={13} fontWeight="600" color="white">{generating ? 'Generating...' : 'Generate'}</Text>
                </XStack>
              </Pressable>
            </YStack>
          </YStack>
        );
    }
  };

  return (
    <SafeAreaView style={{flex: 1, backgroundColor: colors.background}} edges={['top']}>
      <XStack paddingHorizontal={16} paddingVertical={12} alignItems="center" justifyContent="space-between">
        <Text fontSize={20} fontWeight="600" color={colors.text}>Multimodal</Text>
        <Pressable onPress={onRefresh}>
          <Icon name="refresh-cw" size={18} color={colors.primary} />
        </Pressable>
      </XStack>

      {loading ? (
        <YStack flex={1} alignItems="center" justifyContent="center">
          <StatusBadge label="Loading..." variant="info" />
        </YStack>
      ) : (
        <YStack flex={1} paddingHorizontal={16} gap={10}>
          {/* Tabs */}
          <XStack gap={4}>
            {TABS.map(t => (
              <Pressable key={t.key} onPress={() => setTab(t.key)} style={{flex: 1}}>
                <XStack paddingVertical={6} borderRadius={6} backgroundColor={tab === t.key ? colors.primary : 'transparent'} alignItems="center" justifyContent="center" gap={4}>
                  <Icon name={t.icon as any} size={12} color={tab === t.key ? 'white' : colors.textMuted} />
                  <Text fontSize={11} fontWeight={tab === t.key ? '600' : '400'} color={tab === t.key ? 'white' : colors.textMuted}>{t.label}</Text>
                </XStack>
              </Pressable>
            ))}
          </XStack>

          {/* Tab Content */}
          <YStack flex={1}>
            {renderTabContent()}
          </YStack>
        </YStack>
      )}
    </SafeAreaView>
  );
}
