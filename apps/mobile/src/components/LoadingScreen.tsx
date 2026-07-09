import React from 'react';
import {ActivityIndicator} from 'react-native';
import {YStack, Text} from 'tamagui';

export function LoadingScreen() {
  return (
    <YStack flex={1} backgroundColor="$background" alignItems="center" justifyContent="center" gap={16}>
      <Text
        fontSize={48}
        fontWeight="800"
        color="$color9"
        letterSpacing={-1}>
        SG
      </Text>
      <ActivityIndicator size="large" color="$color9" style={{marginTop: 8}} />
      <Text
        fontSize={13}
        fontWeight="400"
        color="$color10"
        lineHeight={18}>
        Loading SloughGPT...
      </Text>
    </YStack>
  );
}
