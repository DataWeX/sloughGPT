import React from 'react';
import {YStack, XStack, Text, useTheme} from 'tamagui';
import {ScreenShell} from '../components/ScreenShell';
import {Icon} from '../components/Icon';

const FEATURES = [
  {icon: 'message-circle', label: 'Real-time chat with streaming'},
  {icon: 'brain', label: 'Model management and switching'},
  {icon: 'zap', label: 'Custom model training (SloNet)'},
  {icon: 'book-open', label: 'Knowledge base management'},
  {icon: 'palette', label: 'Personality (soul) system'},
  {icon: 'bar-chart', label: 'Training metrics and evaluation'},
] as const;

const SHORTCUTS = [
  'Enter \u2014 Send message',
  'Long press \u2014 Message actions',
  'Pull down \u2014 Refresh data',
];

export function AboutScreen() {
  const theme = useTheme();
  const accent = theme.color9?.val || '#7C52C4';

  return (
    <ScreenShell title="About">
      <YStack alignItems="center" paddingVertical={32}>
        <Text fontSize={48} fontWeight="800" color="$color9" letterSpacing={-2}>
          SG
        </Text>
        <Text fontSize={20} fontWeight="600" color="$color" marginTop={8}>
          SloughGPT
        </Text>
        <Text fontSize={13} color="$color10" marginTop={4}>
          v1.0.0
        </Text>
      </YStack>

      <YStack backgroundColor="$background" borderRadius={12} padding={16} borderWidth={0.5} borderColor="$borderColor">
        <Text fontSize={16} fontWeight="600" color="$color" marginBottom={12}>
          About
        </Text>
        <Text fontSize={14} color="$color11" lineHeight={20}>
          SloughGPT is an AI platform that trains and runs custom language models.
          This mobile app connects to your SloughGPT server for chat, model management,
          training, and knowledge management.
        </Text>
      </YStack>

      <YStack backgroundColor="$background" borderRadius={12} padding={16} borderWidth={0.5} borderColor="$borderColor">
        <Text fontSize={16} fontWeight="600" color="$color" marginBottom={12}>
          Features
        </Text>
        <YStack gap={8}>
          {FEATURES.map((f, i) => (
            <XStack key={i} gap={8} alignItems="center">
              <Icon name={f.icon} size={16} color={accent} />
              <Text fontSize={14} color="$color">
                {f.label}
              </Text>
            </XStack>
          ))}
        </YStack>
      </YStack>

      <YStack backgroundColor="$background" borderRadius={12} padding={16} borderWidth={0.5} borderColor="$borderColor">
        <Text fontSize={16} fontWeight="600" color="$color" marginBottom={12}>
          Architecture
        </Text>
        <Text fontSize={14} color="$color11" lineHeight={20}>
          Built with bare React Native CLI. No Expo, no pods. Connects to a Python
          FastAPI backend via REST + SSE streaming. Supports on-device inference
          via ONNX Runtime or llama.rn (planned).
        </Text>
      </YStack>

      <YStack backgroundColor="$background" borderRadius={12} padding={16} borderWidth={0.5} borderColor="$borderColor">
        <Text fontSize={16} fontWeight="600" color="$color" marginBottom={12}>
          Keyboard Shortcuts
        </Text>
        <YStack gap={8}>
          {SHORTCUTS.map((s, i) => (
            <Text key={i} fontSize={14} color="$color">
              {s}
            </Text>
          ))}
        </YStack>
      </YStack>
    </ScreenShell>
  );
}
