import React from 'react';
import {ActivityIndicator} from 'react-native';
import {YStack, Text} from 'tamagui';
import {useColors} from '../theme/colors';

export function LoadingScreen() {
  const c = useColors();
  return (
    <YStack flex={1} backgroundColor={c.background} alignItems="center" justifyContent="center" gap={16}>
      <Text
        fontSize={48}
        fontWeight="800"
        color={c.primary}
        letterSpacing={-1}>
        SG
      </Text>
      <ActivityIndicator size="large" color={c.primary} style={{marginTop: 8}} />
      <Text
        fontSize={13}
        fontWeight="400"
        color={c.textSecondary}
        lineHeight={18}>
        Loading SloughGPT...
      </Text>
    </YStack>
  );
}
