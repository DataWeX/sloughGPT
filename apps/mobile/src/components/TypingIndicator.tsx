/**
 * Typing indicator — animated bouncing dots shown while AI is generating.
 * Appears as a chat bubble with 3 dots that bounce in sequence.
 */

import React, {useEffect, useRef} from 'react';
import {Animated} from 'react-native';
import {XStack} from 'tamagui';

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
    <Animated.View
      style={{
        width: 7,
        height: 7,
        borderRadius: 3.5,
        backgroundColor: '$color10',
        transform: [{translateY: bounce}],
      }}
    />
  );
}

export function TypingIndicator({visible}: Props) {
  if (!visible) return null;

  return (
    <XStack paddingHorizontal={16} marginBottom={8} alignItems="flex-start">
      <XStack
        alignItems="center"
        backgroundColor="$background"
        paddingHorizontal={16}
        paddingVertical={12}
        borderRadius={12}
        borderBottomLeftRadius={4}
        gap={5}>
        <Dot delay={0} />
        <Dot delay={150} />
        <Dot delay={300} />
      </XStack>
    </XStack>
  );
}
