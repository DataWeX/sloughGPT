import React from 'react';
import {YStack, XStack, Text} from 'tamagui';
import {useColors} from '../theme/colors';
import {Icon} from './Icon';

interface CardProps {
  children: React.ReactNode;
  gap?: number;
  noGap?: boolean;
}

export function SettingsCard({children, gap = 12, noGap}: CardProps) {
  const colors = useColors();
  return (
    <YStack
      backgroundColor="$background"
      borderRadius={16}
      borderWidth={1}
      borderColor={colors.border}
      padding={16}
      gap={noGap ? undefined : gap}
      marginBottom={4}
      shadowColor="black"
      shadowOffset={{width: 0, height: 2}}
      shadowOpacity={0.06}
      shadowRadius={8}
      elevation={2}>
      {children}
    </YStack>
  );
}

interface CardHeaderProps {
  icon: string;
  title: string;
}

export function SettingsCardHeader({icon, title}: CardHeaderProps) {
  const colors = useColors();
  return (
    <XStack alignItems="center" gap={10}>
      <YStack width={32} height={32} borderRadius={10} backgroundColor={colors.primaryAlpha(0.1)} alignItems="center" justifyContent="center">
        <Icon name={icon as any} size={16} color={colors.primary} />
      </YStack>
      <Text fontSize={15} fontWeight="600" color="$color">{title}</Text>
    </XStack>
  );
}

interface SelectableChipProps {
  label: string;
  active: boolean;
  onPress: () => void;
}

export function SettingsSelectableChip({label, active, onPress}: SelectableChipProps) {
  const colors = useColors();
  return (
    <YStack
      flex={1}
      paddingVertical={10}
      borderRadius={12}
      backgroundColor={active ? colors.primary : colors.backgroundHover}
      borderWidth={1}
      borderColor={active ? colors.primary : colors.border}
      alignItems="center"
      pressStyle={{opacity: 0.8, scale: 0.97}}
      onPress={onPress}>
      <Text fontSize={12} fontWeight="600" color={active ? 'white' : '$color11'}>
        {label}
      </Text>
    </YStack>
  );
}
