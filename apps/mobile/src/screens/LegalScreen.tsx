import React, {useState} from 'react';
import {FlatList, Pressable, Linking} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {YStack, XStack, Text} from 'tamagui';
import {useColors} from '../theme/colors';
import {Icon} from '../components/Icon';

interface LegalItem {
  id: string;
  title: string;
  description: string;
  icon: string;
  url?: string;
  content?: string;
}

const LEGAL_ITEMS: LegalItem[] = [
  {
    id: 'privacy',
    title: 'Privacy Policy',
    description: 'How SloughGPT collects, uses, and protects your data',
    icon: 'lock',
    content: `SloughGPT Privacy Policy

Last updated: August 20, 2026

1. Data Collection
SloughGPT runs locally on your device or self-hosted server. We do not collect personal data, usage analytics, or device information. All conversations stay on your infrastructure.

2. Push Notifications
If you enable push notifications, your device token is stored on your self-hosted server to deliver notifications. You can disable notifications at any time.

3. Model Training
Training data you provide stays on your server. No data is sent to external services unless you explicitly configure external providers.

4. File Uploads
Files you upload (images, documents, datasets) are processed on your server and are not transmitted to third parties.

5. Open Source
SloughGPT is open source software. You can audit the code at any time.

6. Changes to This Policy
We may update this policy. Changes will be reflected in the app's What's New screen.`,
  },
  {
    id: 'terms',
    title: 'Terms of Service',
    description: 'Usage terms and conditions for SloughGPT',
    icon: 'book',
    content: `SloughGPT Terms of Service

Last updated: August 20, 2026

1. Acceptance
By using SloughGPT, you agree to these terms. SloughGPT is provided as-is for educational and personal use.

2. User Responsibilities
You are responsible for the content you generate using AI models. SloughGPT does not endorse or take responsibility for AI-generated output.

3. Model Usage
AI models may produce inaccurate, biased, or inappropriate content. Always verify important information from authoritative sources.

4. Self-Hosted
When self-hosting, you are responsible for your own data security, backups, and compliance with applicable laws.

5. Limitation of Liability
SloughGPT is provided without warranties. We are not liable for any damages arising from use of the software.

6. Intellectual Property
You retain ownership of content you create. AI models are trained on publicly available datasets.`,
  },
  {
    id: 'open-source',
    title: 'Open Source Licenses',
    description: 'Third-party libraries and their licenses',
    icon: 'layers',
    content: `Third-Party Licenses

React Native - MIT License
Tamagui - MIT License
Zustand - MIT License
Lucide Icons - MIT License
Expo - MIT License
React Navigation - MIT License
Async Storage - MIT License

All third-party libraries used in SloughGPT are open source under permissive licenses. See each library's repository for full license text.`,
  },
  {
    id: 'website',
    title: 'Visit Website',
    description: 'SloughGPT documentation and community',
    icon: 'external-link',
    url: 'https://sloughgpt.app',
  },
  {
    id: 'github',
    title: 'Source Code',
    description: 'View the open source repository',
    icon: 'external-link',
    url: 'https://github.com/sloughgpt/sloughgpt',
  },
];

export function LegalScreen() {
  const colors = useColors();
  const [expandedId, setExpandedId] = useState<string | null>(null);

  return (
    <SafeAreaView style={{flex: 1, backgroundColor: colors.background}} edges={['top']}>
      <XStack paddingHorizontal={16} paddingVertical={12} alignItems="center" justifyContent="space-between">
        <Text fontSize={20} fontWeight="600" color={colors.text}>Legal</Text>
        <Icon name="lock" size={18} color={colors.primary} />
      </XStack>

      <FlatList
        data={LEGAL_ITEMS}
        keyExtractor={item => item.id}
        contentContainerStyle={{paddingHorizontal: 16, paddingBottom: 32, gap: 12}}
        renderItem={({item}) => {
          const isExpanded = expandedId === item.id;
          const hasContent = !!item.content;

          return (
            <Pressable
              onPress={() => {
                if (item.url) {
                  Linking.openURL(item.url).catch(() => {});
                } else if (hasContent) {
                  setExpandedId(isExpanded ? null : item.id);
                }
              }}>
              <YStack
                padding={14}
                borderRadius={10}
                borderWidth={0.5}
                borderColor={isExpanded ? colors.primary : colors.border}
                backgroundColor={colors.white}
                gap={8}>
                <XStack alignItems="center" gap={10}>
                  <YStack
                    width={36} height={36} borderRadius={8}
                    backgroundColor={colors.primaryAlpha(0.1)}
                    alignItems="center" justifyContent="center">
                    <Icon name={item.icon as any} size={16} color={colors.primary} />
                  </YStack>
                  <YStack flex={1} gap={1}>
                    <Text fontSize={14} fontWeight="600" color={colors.text}>{item.title}</Text>
                    <Text fontSize={11} color={colors.textMuted}>{item.description}</Text>
                  </YStack>
                  <Icon
                    name={item.url ? 'external-link' : isExpanded ? 'chevron-up' : 'chevron-down'}
                    size={14}
                    color={colors.textMuted}
                  />
                </XStack>
                {isExpanded && item.content && (
                  <YStack
                    paddingTop={8}
                    borderTopWidth={0.5}
                    borderTopColor={colors.border}>
                    <Text
                      fontSize={12}
                      color={colors.textSecondary}
                      lineHeight={18}
                      selectable>
                      {item.content}
                    </Text>
                  </YStack>
                )}
              </YStack>
            </Pressable>
          );
        }}
      />
    </SafeAreaView>
  );
}
