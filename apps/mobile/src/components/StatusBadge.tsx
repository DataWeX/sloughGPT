import React from 'react';
import {View, Text, StyleSheet} from 'react-native';
import {colors, radii, typography} from '../theme';

interface Props {
  label: string;
  variant?: 'default' | 'success' | 'warning' | 'error' | 'info';
}

const variantColors: Record<string, {bg: string; fg: string}> = {
  default: {bg: colors.surface, fg: colors.textSecondary},
  success: {bg: '#E8F5EE', fg: colors.success},
  warning: {bg: '#FFF3E0', fg: colors.warning},
  error: {bg: '#FDE8E8', fg: colors.error},
  info: {bg: '#EDE7F6', fg: colors.primary},
};

export function StatusBadge({label, variant = 'default'}: Props) {
  const c = variantColors[variant] || variantColors.default;
  return (
    <View style={[styles.badge, {backgroundColor: c.bg}]}>
      <Text style={[styles.text, {color: c.fg}]}>{label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  badge: {
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: radii.full,
    alignSelf: 'flex-start',
  },
  text: {
    ...typography.small,
  },
});
