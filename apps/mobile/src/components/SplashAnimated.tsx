import React, {useEffect, useRef} from 'react';
import {Animated, Dimensions} from 'react-native';
import {YStack, Text} from 'tamagui';
import {useColors} from '../theme/colors';

const {width} = Dimensions.get('window');

export function SplashScreen() {
  const colors = useColors();
  const logoScale = useRef(new Animated.Value(0.5)).current;
  const logoOpacity = useRef(new Animated.Value(0)).current;
  const textOpacity = useRef(new Animated.Value(0)).current;
  const taglineOpacity = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.sequence([
      // Logo fade + scale in
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
      // App name fade in
      Animated.timing(textOpacity, {
        toValue: 1,
        duration: 300,
        useNativeDriver: true,
      }),
      // Tagline fade in
      Animated.timing(taglineOpacity, {
        toValue: 1,
        duration: 300,
        useNativeDriver: true,
      }),
    ]).start();
  }, [logoScale, logoOpacity, textOpacity, taglineOpacity]);

  return (
    <YStack
      flex={1}
      backgroundColor={colors.background}
      alignItems="center"
      justifyContent="center"
      gap={16}>
      {/* Logo circle */}
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

      {/* App name */}
      <Animated.View style={{opacity: textOpacity}}>
        <Text
          fontSize={28}
          fontWeight="700"
          color={colors.text}
          letterSpacing={-0.5}>
          SloughGPT
        </Text>
      </Animated.View>

      {/* Tagline */}
      <Animated.View style={{opacity: taglineOpacity}}>
        <Text
          fontSize={14}
          color={colors.textMuted}
          letterSpacing={0.5}>
          Self-hosted AI assistant
        </Text>
      </Animated.View>
    </YStack>
  );
}
