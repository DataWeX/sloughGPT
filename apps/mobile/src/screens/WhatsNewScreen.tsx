import React, {useState, useEffect, useCallback} from 'react';
import {FlatList} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {YStack, XStack, Text} from 'tamagui';
import {useColors} from '../theme/colors';
import {Icon} from '../components/Icon';
import {StatusBadge} from '../components/StatusBadge';
import AsyncStorage from '@react-native-async-storage/async-storage';

const SEEN_KEY = '@sloughgpt/whats_new_seen';

interface WhatsNewItem {
  id: string;
  title: string;
  description: string;
  icon: string;
  date: string;
  tags?: string[];
}

const ITEMS: WhatsNewItem[] = [
  {
    id: 'deep-linking',
    title: 'Deep linking',
    description: 'Open SloughGPT from links — sloughgpt://chat, sloughgpt://training. Notifications now navigate you to the right screen.',
    icon: '🔗',
    date: '2026-08-20',
    tags: ['Navigation', 'Notifications'],
  },
  {
    id: 'chat-search',
    title: 'Chat search with highlighting',
    description: 'Find text in any conversation. Match count, prev/next navigation, and inline yellow highlighting on matched text.',
    icon: '🔍',
    date: '2026-08-20',
    tags: ['Chat', 'Search'],
  },
  {
    id: 'notification-settings',
    title: 'Notification settings',
    description: 'Per-topic toggles (chat, training), quiet hours, test notification button, and notification history view.',
    icon: '🔔',
    date: '2026-08-20',
    tags: ['Notifications', 'Settings'],
  },
  {
    id: 'import-screen',
    title: 'Import data',
    description: 'Import settings, model checkpoints, training datasets, and soul configurations from files.',
    icon: '📥',
    date: '2026-08-20',
    tags: ['Data', 'Import'],
  },
  {
    id: 'training-push',
    title: 'Training push notifications',
    description: 'Get notified when training completes or fails — no need to watch the progress bar.',
    icon: '📡',
    date: '2026-08-20',
    tags: ['Training', 'Notifications'],
  },
  {
    id: 'memory-screen',
    title: 'Memory management',
    description: 'View, search, store, and consolidate auto-memories. See importance scores and topic distribution.',
    icon: '🧠',
    date: '2026-08-18',
    tags: ['Memory', 'AI'],
  },
  {
    id: 'files-registry',
    title: 'Files & Model Registry',
    description: 'Browse, search, and manage uploaded files. View model registry with health stats and benchmark metrics.',
    icon: '📦',
    date: '2026-08-18',
    tags: ['Files', 'Models'],
  },
  {
    id: 'collections',
    title: 'Data collections',
    description: 'SSE streams, file watchers, generators, and batch pipelines for ingesting and processing training data.',
    icon: '🗄️',
    date: '2026-08-15',
    tags: ['Data', 'Pipeline'],
  },
];

export function WhatsNewScreen() {
  const colors = useColors();
  const [refreshing, setRefreshing] = useState(false);

  const markSeen = useCallback(async () => {
    await AsyncStorage.setItem(SEEN_KEY, new Date().toISOString());
  }, []);

  useEffect(() => {
    markSeen();
  }, [markSeen]);

  return (
    <SafeAreaView style={{flex: 1, backgroundColor: colors.background}} edges={['top']}>
      <XStack paddingHorizontal={16} paddingVertical={12} alignItems="center" justifyContent="space-between">
        <Text fontSize={20} fontWeight="600" color={colors.text}>What's New</Text>
        <Icon name="star" size={18} color={colors.primary} />
      </XStack>

      <FlatList
        data={ITEMS}
        keyExtractor={item => item.id}
        contentContainerStyle={{paddingHorizontal: 16, paddingBottom: 32, gap: 12}}
        renderItem={({item, index}) => (
          <YStack
            padding={14}
            borderRadius={10}
            borderWidth={0.5}
            borderColor={colors.border}
            backgroundColor={colors.white}
            gap={8}>
            <XStack alignItems="center" gap={10}>
              <YStack
                width={36} height={36} borderRadius={8}
                backgroundColor={colors.primaryAlpha(0.1)}
                alignItems="center" justifyContent="center">
                <Text fontSize={18}>{item.icon}</Text>
              </YStack>
              <YStack flex={1} gap={1}>
                <Text fontSize={14} fontWeight="600" color={colors.text}>{item.title}</Text>
                <Text fontSize={10} color={colors.textMuted}>{item.date}</Text>
              </YStack>
              {index === 0 && (
                <StatusBadge label="New" variant="success" />
              )}
            </XStack>
            <Text fontSize={13} color={colors.textSecondary} lineHeight={18}>
              {item.description}
            </Text>
            {item.tags && item.tags.length > 0 && (
              <XStack gap={4} flexWrap="wrap">
                {item.tags.map(tag => (
                  <StatusBadge key={tag} label={tag} variant="default" />
                ))}
              </XStack>
            )}
          </YStack>
        )}
      />
    </SafeAreaView>
  );
}

/** Check if there are unseen items (for badge/dot indicator). */
export async function hasUnseenWhatsNew(): Promise<boolean> {
  const seen = await AsyncStorage.getItem(SEEN_KEY);
  if (!seen) return true;
  const seenDate = new Date(seen).getTime();
  const latestItem = ITEMS[0];
  if (!latestItem) return false;
  const itemDate = new Date(latestItem.date).getTime();
  return itemDate > seenDate;
}
