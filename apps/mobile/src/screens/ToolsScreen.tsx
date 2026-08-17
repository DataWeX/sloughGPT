import React, {useEffect, useState, useCallback} from 'react';
import {Pressable, RefreshControl} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {ScrollView, YStack, XStack, Text, useTheme} from 'tamagui';
import {Icon, type IconName} from '../components/Icon';
import {StatusBadge} from '../components/StatusBadge';
import {useHapticPress} from '../hooks/useHapticPress';
import {useNavigation} from '@react-navigation/native';
import {api} from '../services/api-client';
import type {HealthStatus} from '../types';

interface ToolItem {
  icon: IconName;
  title: string;
  desc: string;
  target: string;
}

const TOOLS: ToolItem[] = [
  {icon: 'dumbbell', title: 'Training', desc: 'Fine-tune models with live loss tracking', target: 'Training'},
  {icon: 'book-open', title: 'Knowledge', desc: "Manage what the AI knows about you", target: 'Knowledge'},
  {icon: 'bookmark', title: 'Bookmarks', desc: 'Saved messages for quick access', target: 'Bookmarks'},
  {icon: 'search', title: 'Search', desc: 'Find messages across conversations', target: 'Search'},
];

function formatUptime(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  const h = Math.floor(seconds / 3600);
  const m = Math.round((seconds % 3600) / 60);
  return `${h}h ${m}m`;
}

export function ToolsScreen() {
  const theme = useTheme();
  const navigation = useNavigation<any>();
  const accent = theme.color9?.val || '#7C52C4';
  const muted = theme.color10?.val || '#827A96';
  const bgCard = theme.background?.val || '#FFFFFF';
  const borderColor = theme.borderColor?.val || '#E4E0F2';
  const hapticPress = useHapticPress();

  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const fetchHealth = useCallback(async () => {
    try {
      const data = await api.get<HealthStatus>('/health');
      setHealth(data);
    } catch {
      setHealth(null);
    }
  }, []);

  useEffect(() => {
    fetchHealth();
  }, [fetchHealth]);

  const onRefresh = async () => {
    setRefreshing(true);
    await fetchHealth();
    setRefreshing(false);
  };

  return (
    <SafeAreaView style={{flex: 1}} edges={['top']}>
      <ScrollView
        flex={1}
        backgroundColor={theme.background?.val || '#F8F6FC'}
        contentContainerStyle={{padding: 16, gap: 12}}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}>
        <Text
          fontSize={24}
          fontWeight="700"
          letterSpacing={-0.3}
          color="$color"
          paddingBottom={4}>
          Tools
        </Text>

        {/* Server Status */}
        <YStack
          backgroundColor={bgCard}
          borderRadius={12}
          borderWidth={0.5}
          borderColor={borderColor}
          padding={16}
          gap={10}>
          <XStack justifyContent="space-between" alignItems="center">
            <Text fontSize={15} fontWeight="600" color="$color">Server</Text>
            <StatusBadge
              label={health?.status === 'healthy' ? 'Connected' : 'Offline'}
              variant={health?.status === 'healthy' ? 'success' : 'error'}
            />
          </XStack>

          {health ? (
            <>
              <XStack justifyContent="space-between" alignItems="center">
                <Text fontSize={13} color="$color11">Model</Text>
                <Text fontSize={13} fontWeight="500" color="$color" numberOfLines={1} maxWidth={180}>
                  {health.model_name || 'None loaded'}
                </Text>
              </XStack>
              <XStack justifyContent="space-between" alignItems="center">
                <Text fontSize={13} color="$color11">Uptime</Text>
                <Text fontSize={13} fontWeight="500" color="$color">{formatUptime(health.uptime)}</Text>
              </XStack>
              <XStack justifyContent="space-between" alignItems="center">
                <Text fontSize={13} color="$color11">Inferences</Text>
                <Text fontSize={13} fontWeight="500" color="$color">{health.inference_count.toLocaleString()}</Text>
              </XStack>
            </>
          ) : (
            <Text fontSize={13} color="$color11" textAlign="center" paddingVertical={8}>
              Could not reach server
            </Text>
          )}

          <Pressable
            onPress={() => hapticPress('light', () => navigation.navigate('Health'))}
            style={({pressed}) => ({opacity: pressed ? 0.6 : 1})}>
            <XStack alignItems="center" justifyContent="center" gap={4} paddingVertical={4}>
              <Text fontSize={12} fontWeight="500" color={accent}>View Details</Text>
              <Icon name="external-link" size={12} color={accent} />
            </XStack>
          </Pressable>
        </YStack>

        {/* Quick Actions */}
        <Text fontSize={13} fontWeight="500" color="$color11" paddingHorizontal={4} marginTop={4}>
          Quick Actions
        </Text>

        {TOOLS.map(item => (
          <Pressable
            key={item.target}
            onPress={() => hapticPress('light', () => navigation.navigate(item.target))}>
            {({pressed}) => (
              <XStack
                backgroundColor={bgCard}
                borderRadius={12}
                borderWidth={0.5}
                borderColor={borderColor}
                padding={14}
                gap={12}
                alignItems="center"
                opacity={pressed ? 0.8 : 1}>
                <YStack
                  width={36}
                  height={36}
                  borderRadius={9}
                  backgroundColor={accent + '12'}
                  alignItems="center"
                  justifyContent="center">
                  <Icon name={item.icon} size={18} color={accent} />
                </YStack>
                <YStack flex={1}>
                  <Text fontSize={14} fontWeight="600" color="$color">{item.title}</Text>
                  <Text fontSize={11} color={muted} marginTop={1}>{item.desc}</Text>
                </YStack>
                <Text fontSize={16} color={muted} fontWeight="300">→</Text>
              </XStack>
            )}
          </Pressable>
        ))}
      </ScrollView>
    </SafeAreaView>
  );
}
