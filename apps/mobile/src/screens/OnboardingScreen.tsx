import React, {useState, useRef, useCallback} from 'react';
import {
  Animated,
  PanResponder,
  Dimensions,
  Pressable,
  TextInput,
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import AsyncStorage from '@react-native-async-storage/async-storage';
import {YStack, XStack, Text, useTheme} from 'tamagui';
import {Icon, type IconName} from '../components/Icon';
import {useHapticPress} from '../hooks/useHapticPress';
import {getApiUrl, setApiUrl} from '../services/api-client';

const {width: SCREEN_W} = Dimensions.get('window');

interface Step {
  icon: IconName;
  title: string;
  desc: string;
  /** When true, this step renders an interactive input instead of static content. */
  interactive?: boolean;
}

const STEPS: Step[] = [
  {
    icon: 'brain',
    title: 'SloughGPT',
    desc: 'Your personal AI platform.\nChat, train, and customize AI models — all from your phone.',
  },
  {
    icon: 'message-circle',
    title: 'Chat with AI',
    desc: 'Send messages and get streaming responses in real time. Swipe left on any message to delete it.',
  },
  {
    icon: 'zap',
    title: 'Train Your Own',
    desc: 'Auto-train custom models from text or datasets. Track progress with live loss charts.',
  },
  {
    icon: 'settings',
    title: 'Connect to Server',
    desc: 'Enter your SloughGPT server URL to get started.',
    interactive: true,
  },
];

const ONBOARD_KEY = '@sloughgpt/onboarded';

export async function isFirstLaunch(): Promise<boolean> {
  const val = await AsyncStorage.getItem(ONBOARD_KEY);
  return val !== 'true';
}

export async function markOnboarded() {
  await AsyncStorage.setItem(ONBOARD_KEY, 'true');
}

interface Props {
  onComplete: () => void;
}

export function OnboardingScreen({onComplete}: Props) {
  const theme = useTheme();
  const [step, setStep] = useState(0);
  const translateX = useRef(new Animated.Value(0)).current;
  const panRef = useRef({startX: 0});

  // Connect step state
  const [serverUrl, setServerUrl] = useState('http://localhost:8000');
  const [connectionStatus, setConnectionStatus] = useState<'idle' | 'testing' | 'ok' | 'fail'>('idle');
  const [connectionError, setConnectionError] = useState('');

  const accent = theme.color9?.val || '#7C52C4';
  const bgBase = theme.background?.val || '#F5F0FF';
  const muted = theme.color10?.val || '#827A96';

  const hapticPress = useHapticPress();

  const testConnection = useCallback(async () => {
    setConnectionStatus('testing');
    setConnectionError('');
    const trimmed = serverUrl.trim();
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 5000);
    try {
      const res = await fetch(`${trimmed}/health`, {signal: controller.signal});
      if (res.ok) {
        await setApiUrl(trimmed);
        setConnectionStatus('ok');
      } else {
        setConnectionStatus('fail');
        setConnectionError(`Server returned ${res.status}`);
      }
    } catch (e: any) {
      setConnectionStatus('fail');
      setConnectionError(e?.name === 'AbortError' ? 'Connection timed out' : e?.message || 'Cannot reach server');
    } finally {
      clearTimeout(timer);
    }
  }, [serverUrl]);

  const panResponder = useRef(
    PanResponder.create({
      onMoveShouldSetPanResponder: (_, g) =>
        Math.abs(g.dx) > 20 && Math.abs(g.dx) > Math.abs(g.dy),
      onPanResponderGrant: () => {
        panRef.current.startX = step * -SCREEN_W;
      },
      onPanResponderMove: (_, g) => {
        const base = panRef.current.startX;
        translateX.setValue(base + g.dx);
      },
      onPanResponderRelease: (_, g) => {
        const target = g.dx < -50 ? (step + 1) * -SCREEN_W : step * -SCREEN_W;
        const nextStep = g.dx < -50 ? Math.min(step + 1, STEPS.length - 1) : step;

        Animated.spring(translateX, {toValue: target, useNativeDriver: true}).start();
        if (nextStep !== step) {
          setStep(nextStep);
          hapticPress('light', () => {});
        }
      },
    }),
  ).current;

  const handleSkip = () => {
    markOnboarded();
    onComplete();
  };

  const handleNext = () => {
    if (step < STEPS.length - 1) {
      const next = step + 1;
      setStep(next);
      Animated.spring(translateX, {toValue: next * -SCREEN_W, useNativeDriver: true}).start();
      hapticPress('light', () => {});
    } else {
      markOnboarded();
      hapticPress('success', () => {});
      onComplete();
    }
  };

  const isLast = step === STEPS.length - 1;
  const currentStep = STEPS[step];

  return (
    <SafeAreaView style={{flex: 1}} edges={['top', 'bottom']}>
      <KeyboardAvoidingView
        style={{flex: 1}}
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
        <YStack flex={1} backgroundColor={bgBase}>
          <Pressable
            style={{position: 'absolute', top: 56, right: 20, zIndex: 10, padding: 8}}
            onPress={handleSkip}>
            <Text fontSize={14} color={muted}>Skip</Text>
          </Pressable>

          <YStack flex={1} overflow="hidden">
            <Animated.View
              style={{
                flexDirection: 'row',
                width: SCREEN_W * STEPS.length,
                flex: 1,
                transform: [{translateX}],
              }}
              {...panResponder.panHandlers}>
              {STEPS.map((s, i) => (
                <YStack
                  key={i}
                  width={SCREEN_W}
                  flex={1}
                  alignItems="center"
                  justifyContent="center"
                  paddingHorizontal={40}>
                  <YStack marginBottom={24}>
                    <Icon name={s.icon} size={48} color={accent} />
                  </YStack>
                  <Text fontSize={24} fontWeight="700" color="$color" marginBottom={12} textAlign="center">
                    {s.title}
                  </Text>
                  <Text fontSize={14} color={muted} textAlign="center" lineHeight={22} marginBottom={s.interactive ? 16 : 0}>
                    {s.desc}
                  </Text>

                  {s.interactive && i === STEPS.length - 1 && (
                    <YStack width="100%" gap={12}>
                      <TextInput
                        value={serverUrl}
                        onChangeText={setServerUrl}
                        placeholder="http://localhost:8000"
                        placeholderTextColor={muted}
                        autoCapitalize="none"
                        autoCorrect={false}
                        keyboardType="url"
                        style={{
                          backgroundColor: theme.background?.val || '#F5F0FF',
                          borderWidth: 1,
                          borderColor: connectionStatus === 'ok'
                            ? '#34B07D'
                            : connectionStatus === 'fail'
                              ? '#DC505A'
                              : theme.borderColor?.val || '#E4E0F2',
                          borderRadius: 10,
                          padding: 14,
                          fontSize: 15,
                          color: theme.color?.val || '#1A1625',
                        }}
                        onSubmitEditing={testConnection}
                        returnKeyType="go"
                      />

                      <Pressable
                        onPress={testConnection}
                        disabled={connectionStatus === 'testing'}
                        style={({pressed}) => ({
                          backgroundColor: accent,
                          paddingVertical: 14,
                          borderRadius: 10,
                          alignItems: 'center',
                          opacity: pressed ? 0.85 : 1,
                        })}>
                        {connectionStatus === 'testing' ? (
                          <ActivityIndicator color="white" />
                        ) : (
                          <Text fontSize={15} fontWeight="600" color="white">
                            {connectionStatus === 'ok' ? 'Connected' : 'Test Connection'}
                          </Text>
                        )}
                      </Pressable>

                      {connectionStatus === 'ok' && (
                        <XStack alignItems="center" justifyContent="center" gap={6}>
                          <Icon name="check" size={16} color="#34B07D" />
                          <Text fontSize={13} color="#34B07D" fontWeight="500">
                            Server is ready
                          </Text>
                        </XStack>
                      )}

                      {connectionStatus === 'fail' && (
                        <Text fontSize={13} color="#DC505A" textAlign="center">
                          {connectionError}
                        </Text>
                      )}
                    </YStack>
                  )}
                </YStack>
              ))}
            </Animated.View>
          </YStack>

          <XStack justifyContent="center" gap={8} paddingVertical={20}>
            {STEPS.map((_, i) => (
              <YStack
                key={i}
                width={i === step ? 24 : 8}
                height={8}
                borderRadius={4}
                backgroundColor={i === step ? accent : theme.borderColor?.val || '#E4E0F2'}
              />
            ))}
          </XStack>

          <Pressable
            style={({pressed}) => ({
              marginHorizontal: 32,
              marginBottom: 48,
              backgroundColor: accent,
              paddingVertical: 16,
              borderRadius: 12,
              alignItems: 'center',
              opacity: pressed ? 0.85 : 1,
            })}
            onPress={handleNext}>
            <Text fontSize={16} fontWeight="600" color="white">
              {isLast ? 'Get Started' : 'Next'}
            </Text>
          </Pressable>
        </YStack>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}
