import React from 'react';
import {View, Text, ScrollView, StyleSheet, Linking, TouchableOpacity} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {colors, spacing, radii, typography} from '../theme';

export function AboutScreen() {
  return (
    <SafeAreaView style={styles.safe} edges={['top']}>
      <ScrollView contentContainerStyle={styles.content}>
        <View style={styles.logoContainer}>
          <Text style={styles.logo}>SG</Text>
          <Text style={styles.appName}>SloughGPT</Text>
          <Text style={styles.version}>v1.0.0</Text>
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>About</Text>
          <Text style={styles.text}>
            SloughGPT is an AI platform that trains and runs custom language models.
            This mobile app connects to your SloughGPT server for chat, model management,
            training, and knowledge management.
          </Text>
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Features</Text>
          <Text style={styles.listItem}>💬 Real-time chat with streaming</Text>
          <Text style={styles.listItem}>🧠 Model management and switching</Text>
          <Text style={styles.listItem}>🏋️ Custom model training (SloNet)</Text>
          <Text style={styles.listItem}>📚 Knowledge base management</Text>
          <Text style={styles.listItem}>🎨 Personality (soul) system</Text>
          <Text style={styles.listItem}>📊 Training metrics and evaluation</Text>
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Architecture</Text>
          <Text style={styles.text}>
            Built with bare React Native CLI. No Expo, no pods. Connects to a Python
            FastAPI backend via REST + SSE streaming. Supports on-device inference
            via ONNX Runtime or llama.rn (planned).
          </Text>
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Keyboard Shortcuts</Text>
          <Text style={styles.listItem}>Enter — Send message</Text>
          <Text style={styles.listItem}>Long press — Message actions</Text>
          <Text style={styles.listItem}>Pull down — Refresh data</Text>
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
  content: {
    padding: spacing.lg,
    gap: spacing.md,
  },
  logoContainer: {
    alignItems: 'center',
    paddingVertical: spacing.xxxl,
  },
  logo: {
    fontSize: 56,
    fontWeight: '800',
    color: colors.primary,
    letterSpacing: -2,
  },
  appName: {
    ...typography.h1,
    color: colors.text,
    marginTop: spacing.sm,
  },
  version: {
    ...typography.caption,
    color: colors.textMuted,
    marginTop: spacing.xs,
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
  text: {
    ...typography.body,
    color: colors.textSecondary,
  },
  listItem: {
    ...typography.body,
    color: colors.text,
    paddingVertical: spacing.xs,
  },
});
