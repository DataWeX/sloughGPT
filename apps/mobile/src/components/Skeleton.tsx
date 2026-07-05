/**
 * Skeleton loading placeholder — animated pulse effect.
 * Reusable for any loading state in the app.
 */

import React, {useEffect, useRef} from 'react';
import {View, Animated, StyleSheet} from 'react-native';
import {colors, radii} from '../theme';

interface SkeletonProps {
  width?: number | string;
  height?: number;
  borderRadius?: number;
  style?: object;
}

export function Skeleton({width, height = 16, borderRadius = radii.sm, style}: SkeletonProps) {
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
        {width, height, borderRadius, backgroundColor: colors.border, opacity: pulse},
        style,
      ]}
    />
  );
}

/** Pre-built skeleton patterns for common layouts */

export function SkeletonCard({lines = 3}: {lines?: number}) {
  return (
    <View style={cardStyles.card}>
      <Skeleton width="40%" height={14} />
      <View style={{gap: 8, marginTop: 12}}>
        {Array.from({length: lines}).map((_, i) => (
          <Skeleton
            key={i}
            width={i === lines - 1 ? '60%' : '100%'}
            height={12}
          />
        ))}
      </View>
    </View>
  );
}

export function SkeletonChatBubble() {
  return (
    <View style={chatStyles.row}>
      <Skeleton width="70%" height={40} borderRadius={16} />
    </View>
  );
}

export function SkeletonList({count = 4, lines = 2}: {count?: number; lines?: number}) {
  return (
    <View style={listStyles.container}>
      {Array.from({length: count}).map((_, i) => (
        <SkeletonCard key={i} lines={lines} />
      ))}
    </View>
  );
}

const cardStyles = StyleSheet.create({
  card: {
    backgroundColor: colors.surface,
    borderRadius: radii.md,
    padding: 16,
    marginBottom: 12,
  },
});

const chatStyles = StyleSheet.create({
  row: {
    paddingHorizontal: 16,
    marginBottom: 12,
    alignItems: 'flex-start',
  },
});

const listStyles = StyleSheet.create({
  container: {
    padding: 16,
  },
});
