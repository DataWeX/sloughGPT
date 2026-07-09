import React from 'react';
import {ScrollView, KeyboardAvoidingView, Platform, RefreshControl} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {YStack, Text} from 'tamagui';

interface ScreenShellProps {
  title: string;
  children: React.ReactNode;
  scroll?: boolean;
  backgroundColor?: string;
  refreshing?: boolean;
  onRefresh?: () => void;
}

export function ScreenShell({
  title,
  children,
  scroll = true,
  backgroundColor = '#F5F0FF',
  refreshing,
  onRefresh,
}: ScreenShellProps) {
  const Wrapper = scroll ? ScrollView : React.Fragment;
  const wrapperProps: any = scroll
    ? {
        contentContainerStyle: {padding: 16, gap: 12},
        ...(onRefresh ? {refreshControl: <RefreshControl refreshing={!!refreshing} onRefresh={onRefresh} />} : {}),
      }
    : {};

  return (
    <SafeAreaView style={{flex: 1, backgroundColor}} edges={['top']}>
      <KeyboardAvoidingView
        style={{flex: 1}}
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
        <YStack flex={1}>
          <Text
            fontSize={24}
            fontWeight="700"
            letterSpacing={-0.3}
            color="$color"
            paddingHorizontal={16}
            paddingTop={4}
            paddingBottom={4}>
            {title}
          </Text>
          <Wrapper {...wrapperProps}>{children}</Wrapper>
        </YStack>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}
