/**
 * Typing indicator — animated bouncing dots shown while AI is generating.
 * Appears as a chat bubble with 3 dots that bounce in sequence.
 */

import React, {useEffect, useRef} from 'react';
import {View, Animated, StyleSheet} from 'react-native';
import {colors, radii, spacing} from '../theme';

interface Props {
  visible: boolean;
}

function Dot({delay}: {delay: number}) {
  const bounce = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    const loop = Animated.loop(
      Animated.sequence([
        Animated.delay(delay),
        Animated.timing(bounce, {toValue: -6, duration: 200, useNativeDriver: true}),
        Animated.timing(bounce, {toValue: 0, duration: 200, useNativeDriver: true}),
      ]),
    );
    loop.start();
    return () => loop.stop();
  }, []);

  return (
    <Animated.View style={[styles.dot, {transform: [{translateY: bounce}]}]} />
  );
}

export function TypingIndicator({visible}: Props) {
  if (!visible) return null;

  return (
    <View style={styles.row}>
      <View style={styles.bubble}>
        <Dot delay={0} />
        <Dot delay={150} />
        <Dot delay={300} />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  row: {
    paddingHorizontal: spacing.lg,
    marginBottom: spacing.sm,
    alignItems: 'flex-start',
  },
  bubble: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.surface,
    paddingHorizontal: spacing.md + 4,
    paddingVertical: spacing.sm + 4,
    borderRadius: radii.lg,
    borderBottomLeftRadius: radii.sm,
    gap: 5,
  },
  dot: {
    width: 7,
    height: 7,
    borderRadius: 3.5,
    backgroundColor: colors.textMuted,
  },
});
