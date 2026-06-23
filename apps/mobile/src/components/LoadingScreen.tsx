import React from 'react';
import {View, Text, StyleSheet, ActivityIndicator} from 'react-native';
import {colors, spacing, typography} from '../theme';

export function LoadingScreen() {
  return (
    <View style={styles.container}>
      <Text style={styles.logo}>SG</Text>
      <ActivityIndicator size="large" color={colors.primary} style={styles.spinner} />
      <Text style={styles.text}>Loading SloughGPT...</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.lg,
  },
  logo: {
    fontSize: 48,
    fontWeight: '800',
    color: colors.primary,
    letterSpacing: -1,
  },
  spinner: {
    marginTop: spacing.sm,
  },
  text: {
    ...typography.caption,
    color: colors.textMuted,
  },
});
