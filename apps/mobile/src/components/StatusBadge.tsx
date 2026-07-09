import React from 'react';
import {Text} from 'tamagui';

interface Props {
  label: string;
  variant?: 'default' | 'success' | 'warning' | 'error' | 'info';
}

const variantColors: Record<string, {bg: string; fg: string}> = {
  default: {bg: '$background', fg: '$color11'},
  success: {bg: '#E8F5EE', fg: '#22C55E'},
  warning: {bg: '#FFF3E0', fg: '#F59E0B'},
  error: {bg: '#FDE8E8', fg: '#EF4444'},
  info: {bg: '$background', fg: '$color9'},
};

export function StatusBadge({label, variant = 'default'}: Props) {
  const c = variantColors[variant] || variantColors.default;
  return (
    <Text
      fontSize={11}
      fontWeight="500"
      letterSpacing={0.2}
      paddingHorizontal={8}
      paddingVertical={3}
      borderRadius={9999}
      alignSelf="flex-start"
      backgroundColor={c.bg}
      color={c.fg}
      overflow="hidden">
      {label}
    </Text>
  );
}
