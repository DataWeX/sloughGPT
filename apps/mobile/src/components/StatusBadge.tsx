import React from 'react';
import {Text} from 'tamagui';
import {useColors} from '../theme/colors';

interface Props {
  label: string;
  variant?: 'default' | 'success' | 'warning' | 'error' | 'info';
}

function getVariantColors(variant: string, c: ReturnType<typeof useColors>) {
  switch (variant) {
    case 'success':
      return {bg: c.successLight, fg: c.success};
    case 'warning':
      return {bg: c.warning + '20', fg: c.warning};
    case 'error':
      return {bg: c.errorLight, fg: c.error};
    case 'info':
      return {bg: c.background, fg: c.primary};
    default:
      return {bg: c.background, fg: c.textMuted};
  }
}

export function StatusBadge({label, variant = 'default'}: Props) {
  const c = useColors();
  const colors = getVariantColors(variant, c);
  return (
    <Text
      fontSize={11}
      fontWeight="500"
      letterSpacing={0.2}
      paddingHorizontal={8}
      paddingVertical={3}
      borderRadius={9999}
      alignSelf="flex-start"
      backgroundColor={colors.bg}
      color={colors.fg}
      overflow="hidden">
      {label}
    </Text>
  );
}
