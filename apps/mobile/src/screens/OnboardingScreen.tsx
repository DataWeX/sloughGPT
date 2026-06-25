import React, {useState} from 'react';
import {
  View,
  Text,
  ScrollView,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  Dimensions,
} from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import {colors, spacing, radii, typography} from '../theme';
import {api, setApiUrl} from '../services/api-client';

const {width} = Dimensions.get('window');

const STEPS = [
  {
    icon: '🧠',
    title: 'Welcome to SloughGPT',
    desc: 'Train and run custom AI models from your phone.',
  },
  {
    icon: '🔌',
    title: 'Connect to Server',
    desc: 'Enter your SloughGPT server URL to get started.',
  },
  {
    icon: '💬',
    title: 'Start Chatting',
    desc: 'Load a model, pick a personality, and start chatting.',
  },
  {
    icon: '🏋️',
    title: 'Train Your Own',
    desc: 'Paste text or pick a dataset to train a custom model.',
  },
];

export function OnboardingScreen({onDone}: {onDone: () => void}) {
  const [step, setStep] = useState(0);
  const [serverUrl, setServerUrl] = useState('http://localhost:8000');
  const [connecting, setConnecting] = useState(false);
  const [connected, setConnected] = useState<boolean | null>(null);

  const handleConnect = async () => {
    setConnecting(true);
    setConnected(null);
    try {
      const url = serverUrl.trim();
      await setApiUrl(url);
      const res = await fetch(url + '/health');
      setConnected(res.ok);
    } catch {
      setConnected(false);
    }
    setConnecting(false);
  };

  const handleFinish = async () => {
    await AsyncStorage.setItem('@sloughgpt/onboarded', 'true');
    onDone();
  };

  return (
    <View style={styles.container}>
      <ScrollView
        horizontal
        pagingEnabled
        showsHorizontalScrollIndicator={false}
        contentOffset={{x: step * width, y: 0}}
        scrollEnabled={false}>
        {STEPS.map((s, i) => (
          <View key={i} style={[styles.page, {width}]}>
            <Text style={styles.icon}>{s.icon}</Text>
            <Text style={styles.title}>{s.title}</Text>
            <Text style={styles.desc}>{s.desc}</Text>

            {i === 1 && (
              <View style={styles.serverSection}>
                <TextInput
                  style={styles.urlInput}
                  value={serverUrl}
                  onChangeText={setServerUrl}
                  placeholder="http://localhost:8000"
                  placeholderTextColor={colors.textMuted}
                  autoCapitalize="none"
                  autoCorrect={false}
                  keyboardType="url"
                />
                <TouchableOpacity
                  style={[styles.connectBtn, connecting && styles.connectBtnLoading]}
                  onPress={handleConnect}
                  disabled={connecting}>
                  <Text style={styles.connectBtnText}>
                    {connecting ? 'Connecting...' : connected === true ? 'Connected ✓' : 'Connect'}
                  </Text>
                </TouchableOpacity>
                {connected === false && (
                  <Text style={styles.connectError}>
                    Could not connect. Make sure the server is running.
                  </Text>
                )}
              </View>
            )}
          </View>
        ))}
      </ScrollView>

      <View style={styles.footer}>
        <View style={styles.dots}>
          {STEPS.map((_, i) => (
            <View
              key={i}
              style={[styles.dot, i === step && styles.dotActive]}
            />
          ))}
        </View>
        <View style={styles.btnRow}>
          {step > 0 && (
            <TouchableOpacity style={styles.backBtn} onPress={() => setStep(s => s - 1)}>
              <Text style={styles.backBtnText}>Back</Text>
            </TouchableOpacity>
          )}
          <TouchableOpacity
            style={[styles.nextBtn, step === STEPS.length - 1 && connected === false && styles.nextBtnDisabled]}
            onPress={() => {
              if (step < STEPS.length - 1) {
                setStep(s => s + 1);
              } else {
                handleFinish();
              }
            }}
            disabled={step === STEPS.length - 1 && connected === false}>
            <Text style={styles.nextBtnText}>
              {step === STEPS.length - 1 ? 'Get Started' : 'Next'}
            </Text>
          </TouchableOpacity>
        </View>
        {step === STEPS.length - 1 && (
          <TouchableOpacity onPress={handleFinish}>
            <Text style={styles.skipText}>Skip for now</Text>
          </TouchableOpacity>
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
  },
  page: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: spacing.xxxl,
    paddingBottom: 120,
  },
  icon: {
    fontSize: 64,
    marginBottom: spacing.xxl,
  },
  title: {
    ...typography.h1,
    color: colors.text,
    textAlign: 'center',
    marginBottom: spacing.md,
  },
  desc: {
    ...typography.body,
    color: colors.textSecondary,
    textAlign: 'center',
    lineHeight: 24,
  },
  serverSection: {
    width: '100%',
    marginTop: spacing.xxl,
    gap: spacing.sm,
  },
  urlInput: {
    ...typography.body,
    color: colors.text,
    backgroundColor: colors.surface,
    borderRadius: radii.md,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
  },
  connectBtn: {
    backgroundColor: colors.primary,
    paddingVertical: spacing.md,
    borderRadius: radii.md,
    alignItems: 'center',
  },
  connectBtnLoading: {
    opacity: 0.6,
  },
  connectBtnText: {
    ...typography.body,
    color: colors.white,
    fontWeight: '600',
  },
  connectError: {
    ...typography.caption,
    color: colors.error,
    textAlign: 'center',
  },
  footer: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    paddingHorizontal: spacing.xxxl,
    paddingBottom: spacing.xxxl,
    gap: spacing.lg,
  },
  dots: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: spacing.sm,
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
  btnRow: {
    flexDirection: 'row',
    gap: spacing.sm,
  },
  backBtn: {
    flex: 1,
    paddingVertical: spacing.md,
    borderRadius: radii.md,
    alignItems: 'center',
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.border,
  },
  backBtnText: {
    ...typography.body,
    color: colors.textSecondary,
    fontWeight: '600',
  },
  nextBtn: {
    flex: 2,
    paddingVertical: spacing.md,
    borderRadius: radii.md,
    alignItems: 'center',
    backgroundColor: colors.primary,
  },
  nextBtnDisabled: {
    opacity: 0.4,
  },
  nextBtnText: {
    ...typography.body,
    color: colors.white,
    fontWeight: '600',
  },
  skipText: {
    ...typography.caption,
    color: colors.textMuted,
    textAlign: 'center',
  },
});
