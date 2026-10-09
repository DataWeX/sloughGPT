import React, {useEffect, useRef} from 'react';
import {Animated, View as RNView} from 'react-native';
import {YStack, Text} from 'tamagui';
import {useColors} from '../theme/colors';

export function SplashScreen({onDone}: {onDone?: () => void}) {
  const colors = useColors();
  const logoScale = useRef(new Animated.Value(0.5)).current;
  const logoOpacity = useRef(new Animated.Value(0)).current;
  const textOpacity = useRef(new Animated.Value(0)).current;
  const taglineOpacity = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.sequence([
      Animated.parallel([
        Animated.spring(logoScale, {
          toValue: 1,
          friction: 6,
          tension: 40,
          useNativeDriver: true,
        }),
        Animated.timing(logoOpacity, {
          toValue: 1,
          duration: 400,
          useNativeDriver: true,
        }),
      ]),
      Animated.timing(textOpacity, {
        toValue: 1,
        duration: 300,
        useNativeDriver: true,
      }),
      Animated.timing(taglineOpacity, {
        toValue: 1,
        duration: 300,
        useNativeDriver: true,
      }),
    ]).start();

    const timer = setTimeout(() => onDone?.(), 1200);
    return () => clearTimeout(timer);
  }, [logoScale, logoOpacity, textOpacity, taglineOpacity, onDone]);

  return (
    <RNView style={{flex: 1, backgroundColor: colors.background}}>
      <YStack
        flex={1}
        alignItems="center"
        justifyContent="center"
        gap={16}>
        <Animated.View
          style={{
            opacity: logoOpacity,
            transform: [{scale: logoScale}],
          }}>
          <YStack
            width={100}
            height={100}
            borderRadius={50}
            backgroundColor={colors.primaryAlpha(0.12)}
            alignItems="center"
            justifyContent="center"
            borderWidth={1}
            borderColor={colors.primaryAlpha(0.2)}>
            <YStack
              width={72}
              height={72}
              borderRadius={36}
              backgroundColor={colors.primaryAlpha(0.15)}
              alignItems="center"
              justifyContent="center">
              <Text
                fontSize={32}
                fontWeight="800"
                color={colors.primary}
                letterSpacing={-1}>
                SG
              </Text>
            </YStack>
          </YStack>
        </Animated.View>

        <Animated.View style={{opacity: textOpacity}}>
          <Text
            fontSize={28}
            fontWeight="700"
            color={colors.text}
            letterSpacing={-0.5}>
            SloughGPT
          </Text>
        </Animated.View>

        <Animated.View style={{opacity: taglineOpacity}}>
          <Text
            fontSize={14}
            color={colors.textMuted}
            letterSpacing={0.5}>
            Self-hosted AI assistant
          </Text>
        </Animated.View>
      </YStack>
    </RNView>
  );
}
