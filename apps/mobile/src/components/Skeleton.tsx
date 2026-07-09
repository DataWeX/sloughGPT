/**
 * Skeleton loading placeholder — animated pulse effect.
 * Reusable for any loading state in the app.
 */

import React, {useEffect, useRef} from 'react';
import {Animated} from 'react-native';
import {YStack, XStack} from 'tamagui';

interface SkeletonProps {
  width?: number | string;
  height?: number;
  borderRadius?: number;
  style?: object;
}

export function Skeleton({width, height = 16, borderRadius = 4, style}: SkeletonProps) {
  const pulse = useRef(new Animated.Value(0.3)).current;

  useEffect(() => {
    const loop = Animated.loop(
      Animated.sequence([
        Animated.timing(pulse, {toValue: 0.6, duration: 800, useNativeDriver: true}),
        Animated.timing(pulse, {toValue: 0.3, duration: 800, useNativeDriver: true}),
      ]),
    );
    loop.start();
    return () => loop.stop();
  }, []);

  return (
    <Animated.View
      style={[
        {width, height, borderRadius, backgroundColor: '$borderColor', opacity: pulse},
        style,
      ]}
    />
  );
}

/** Pre-built skeleton patterns for common layouts */

export function SkeletonCard({lines = 3}: {lines?: number}) {
  return (
    <YStack backgroundColor="$background" borderRadius={8} padding={16} marginBottom={12}>
      <Skeleton width="40%" height={14} />
      <YStack gap={8} marginTop={12}>
        {Array.from({length: lines}).map((_, i) => (
          <Skeleton
            key={i}
            width={i === lines - 1 ? '60%' : '100%'}
            height={12}
          />
        ))}
      </YStack>
    </YStack>
  );
}

export function SkeletonChatBubble() {
  return (
    <XStack paddingHorizontal={16} marginBottom={12} alignItems="flex-start">
      <Skeleton width="70%" height={40} borderRadius={16} />
    </XStack>
  );
}

export function SkeletonList({count = 4, lines = 2}: {count?: number; lines?: number}) {
  return (
    <YStack padding={16}>
      {Array.from({length: count}).map((_, i) => (
        <SkeletonCard key={i} lines={lines} />
      ))}
    </YStack>
  );
}
