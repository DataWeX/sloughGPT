import React from 'react';
import {Pressable} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {YStack, XStack, Text, useTheme} from 'tamagui';
import {Icon, type IconName} from '../components/Icon';
import {useHapticPress} from '../hooks/useHapticPress';
import {useNavigation} from '@react-navigation/native';

interface ToolItem {
  icon: IconName;
  title: string;
  desc: string;
  target: string;
}

const TOOLS: ToolItem[] = [
  {icon: 'dumbbell', title: 'Training', desc: 'Fine-tune models and track progress', target: 'Training'},
  {icon: 'book-open', title: 'Knowledge', desc: "View and manage what the AI knows about you", target: 'Knowledge'},
  {icon: 'bookmark', title: 'Bookmarks', desc: 'Saved messages for quick access', target: 'Bookmarks'},
  {icon: 'search', title: 'Search', desc: 'Find messages across conversations', target: 'Search'},
  {icon: 'target', title: 'Health', desc: 'Server status and system metrics', target: 'Health'},
];

export function ToolsScreen() {
  const theme = useTheme();
  const navigation = useNavigation<any>();
  const accent = theme.color9?.val || '#7C52C4';
  const muted = theme.color10?.val || '#827A96';
  const hapticPress = useHapticPress();

  return (
    <SafeAreaView style={{flex: 1}} edges={['top']}>
      <YStack flex={1} backgroundColor={theme.background?.val || '#F8F6FC'}>
        <Text
          fontSize={24}
          fontWeight="700"
          letterSpacing={-0.3}
          color="$color"
          paddingHorizontal={16}
          paddingTop={4}
          paddingBottom={4}>
          Tools
        </Text>

        <YStack flex={1} paddingHorizontal={16} paddingTop={8} gap={10}>
          {TOOLS.map(item => (
            <Pressable
              key={item.target}
              onPress={() => hapticPress('light', () => navigation.navigate(item.target))}>
              {({pressed}) => (
                <XStack
                  backgroundColor={theme.background?.val || '#FFFFFF'}
                  borderRadius={12}
                  borderWidth={0.5}
                  borderColor={theme.borderColor?.val || '#E4E0F2'}
                  padding={16}
                  gap={14}
                  alignItems="center"
                  opacity={pressed ? 0.8 : 1}>
                  <YStack
                    width={40}
                    height={40}
                    borderRadius={10}
                    backgroundColor={accent + '15'}
                    alignItems="center"
                    justifyContent="center">
                    <Icon name={item.icon} size={20} color={accent} />
                  </YStack>
                  <YStack flex={1}>
                    <Text fontSize={15} fontWeight="600" color="$color">{item.title}</Text>
                    <Text fontSize={12} color={muted} marginTop={2}>{item.desc}</Text>
                  </YStack>
                  <Text fontSize={18} color={muted} fontWeight="300">→</Text>
                </XStack>
              )}
            </Pressable>
          ))}
        </YStack>
      </YStack>
    </SafeAreaView>
  );
}
