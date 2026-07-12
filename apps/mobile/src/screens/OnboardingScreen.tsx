import React, {useState, useRef} from 'react';
import {
  Animated,
  PanResponder,
  Dimensions,
  Pressable,
} from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import {YStack, XStack, Text, useTheme} from 'tamagui';
import {triggerHaptic} from '../services/haptics';
import {Icon, type IconName} from '../components/Icon';

const {width: SCREEN_W} = Dimensions.get('window');

const STEPS: {icon: IconName; title: string; desc: string}[] = [
  {
    icon: 'message-circle',
    title: 'Chat with AI',
    desc: 'Send messages and get streaming responses. Swipe left on any message to delete it.',
  },
  {
    icon: 'brain',
    title: 'Switch Personalities',
    desc: 'Choose different AI souls in Models — each has a unique personality and style.',
  },
  {
    icon: 'zap',
    title: 'Train Your Own',
    desc: 'Auto-train custom models from text or datasets. Track progress with live loss charts.',
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

  const accent = theme.color9?.val || '#7C52C4';
  const bgBase = theme.background?.val || '#F5F0FF';

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
        const base = panRef.current.startX;
        const target = g.dx < -50 ? (step + 1) * -SCREEN_W : step * -SCREEN_W;
        const nextStep = g.dx < -50 ? Math.min(step + 1, STEPS.length - 1) : step;

        Animated.spring(translateX, {toValue: target, useNativeDriver: true}).start();
        if (nextStep !== step) {
          setStep(nextStep);
          triggerHaptic('light');
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
      triggerHaptic('light');
    } else {
      markOnboarded();
      triggerHaptic('success');
      onComplete();
    }
  };

  return (
    <YStack flex={1} backgroundColor={bgBase}>
      <Pressable
        style={{position: 'absolute', top: 56, right: 20, zIndex: 10, padding: 8}}
        onPress={handleSkip}>
        <Text fontSize={14} color="$color10">Skip</Text>
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
              <Text fontSize={14} color="$color10" textAlign="center" lineHeight={22}>
                {s.desc}
              </Text>
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
            backgroundColor={i === step ? '$color9' : '$borderColor'}
          />
        ))}
      </XStack>

      <Pressable
        style={{
          marginHorizontal: 32,
          marginBottom: 48,
          backgroundColor: accent,
          paddingVertical: 16,
          borderRadius: 12,
          alignItems: 'center',
        }}
        onPress={handleNext}>
        <Text fontSize={16} fontWeight="600" color="white">
          {step < STEPS.length - 1 ? 'Next' : 'Get Started'}
        </Text>
      </Pressable>
    </YStack>
  );
}
