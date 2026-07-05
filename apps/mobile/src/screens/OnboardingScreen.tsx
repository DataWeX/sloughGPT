/**
 * First-launch onboarding flow — 3 swipeable cards explaining key features.
 * Shows once, then stored in AsyncStorage. Accessible from Settings.
 */

import React, {useState, useRef} from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Animated,
  PanResponder,
  Dimensions,
} from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import {triggerHaptic} from '../services/haptics';
import {colors, radii, typography} from '../theme';

const {width: SCREEN_W} = Dimensions.get('window');

const STEPS = [
  {
    icon: '💬',
    title: 'Chat with AI',
    desc: 'Send messages and get streaming responses. Swipe left on any message to delete it.',
  },
  {
    icon: '🧠',
    title: 'Switch Personalities',
    desc: 'Choose different AI souls in Models — each has a unique personality and style.',
  },
  {
    icon: '🏋️',
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
  const [step, setStep] = useState(0);
  const translateX = useRef(new Animated.Value(0)).current;
  const panRef = useRef({startX: 0});

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
    <View style={styles.container}>
      {/* Skip */}
      <TouchableOpacity style={styles.skipBtn} onPress={handleSkip}>
        <Text style={styles.skipText}>Skip</Text>
      </TouchableOpacity>

      {/* Cards */}
      <View style={styles.cardsWrap}>
        <Animated.View
          style={[styles.cardsRow, {transform: [{translateX}]}]}
          {...panResponder.panHandlers}>
          {STEPS.map((s, i) => (
            <View key={i} style={styles.card}>
              <Text style={styles.icon}>{s.icon}</Text>
              <Text style={styles.title}>{s.title}</Text>
              <Text style={styles.desc}>{s.desc}</Text>
            </View>
          ))}
        </Animated.View>
      </View>

      {/* Dots */}
      <View style={styles.dots}>
        {STEPS.map((_, i) => (
          <View
            key={i}
            style={[styles.dot, i === step && styles.dotActive]}
          />
        ))}
      </View>

      {/* CTA */}
      <TouchableOpacity style={styles.cta} onPress={handleNext} activeOpacity={0.8}>
        <Text style={styles.ctaText}>
          {step < STEPS.length - 1 ? 'Next' : 'Get Started'}
        </Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
  },
  skipBtn: {
    position: 'absolute',
    top: 56,
    right: 20,
    zIndex: 10,
    padding: 8,
  },
  skipText: {
    ...typography.body,
    color: colors.textMuted,
  },
  cardsWrap: {
    flex: 1,
    overflow: 'hidden',
  },
  cardsRow: {
    flexDirection: 'row',
    width: SCREEN_W * STEPS.length,
    flex: 1,
  },
  card: {
    width: SCREEN_W,
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 40,
  },
  icon: {
    fontSize: 64,
    marginBottom: 24,
  },
  title: {
    fontSize: 24,
    fontWeight: '700',
    color: colors.text,
    marginBottom: 12,
    textAlign: 'center',
  },
  desc: {
    ...typography.body,
    color: colors.textMuted,
    textAlign: 'center',
    lineHeight: 22,
  },
  dots: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 8,
    paddingVertical: 20,
  },
  dot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: colors.border,
  },
  dotActive: {
    backgroundColor: colors.primary,
    width: 24,
  },
  cta: {
    marginHorizontal: 32,
    marginBottom: 48,
    backgroundColor: colors.primary,
    paddingVertical: 16,
    borderRadius: radii.md,
    alignItems: 'center',
  },
  ctaText: {
    ...typography.body,
    color: colors.white,
    fontWeight: '600',
    fontSize: 16,
  },
});
