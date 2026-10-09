import React, {useEffect, useState, useCallback} from 'react';
import {Pressable, RefreshControl, ScrollView} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {YStack, XStack, Text} from 'tamagui';
import {useColors} from '../theme/colors';
import {useModelStore} from '../stores/model-store';
import {useChatStore} from '../stores/chat-store';
import {api} from '../services/api-client';
import {Icon} from '../components/Icon';
import {StatusBadge} from '../components/StatusBadge';
import {triggerHaptic} from '../services/haptics';
import {useSidebar} from '../contexts/SidebarContext';
import type {Session} from '../types';

interface SystemHealth {
  model_loaded: boolean;
  model_name: string | null;
  uptime_s: number;
  request_count: number;
  error_count: number;
}

function formatUptime(seconds: number): string {
  if (seconds < 60) return `${Math.floor(seconds)}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  return `${h}h ${m}m`;
}

function QuickAction({icon, label, onPress}: {icon: string; label: string; onPress: () => void}) {
  const colors = useColors();
  return (
    <Pressable onPress={onPress}>
      {({pressed}) => (
        <YStack
          flex={1}
          alignItems="center"
          gap={8}
          padding={14}
          borderRadius={14}
          backgroundColor={pressed ? colors.primaryAlpha(0.08) : colors.backgroundHover}
          borderWidth={1}
          borderColor={colors.border}
          pressStyle={{opacity: 0.8, scale: 0.97}}>
          <YStack
            width={44}
            height={44}
            borderRadius={14}
            backgroundColor={colors.primaryAlpha(0.12)}
            alignItems="center"
            justifyContent="center">
            <Icon name={icon as any} size={20} color={colors.primary} />
          </YStack>
          <Text fontSize={12} fontWeight="600" color={colors.text} textAlign="center">
            {label}
          </Text>
        </YStack>
      )}
    </Pressable>
  );
}

function Card({children, ...props}: {children: React.ReactNode;[key: string]: any}) {
  const colors = useColors();
  return (
    <YStack
      backgroundColor="$background"
      borderRadius={16}
      borderWidth={1}
      borderColor={colors.border}
      padding={16}
      gap={12}
      shadowColor="black"
      shadowOffset={{width: 0, height: 2}}
      shadowOpacity={0.06}
      shadowRadius={8}
      elevation={2}
      {...props}>
      {children}
    </YStack>
  );
}

function SectionHeader({icon, title}: {icon: string; title: string}) {
  const colors = useColors();
  return (
    <XStack alignItems="center" gap={10}>
      <YStack width={32} height={32} borderRadius={10} backgroundColor={colors.primaryAlpha(0.1)} alignItems="center" justifyContent="center">
        <Icon name={icon as any} size={16} color={colors.primary} />
      </YStack>
      <Text fontSize={15} fontWeight="600" color="$color">{title}</Text>
    </XStack>
  );
}

export function HomeScreen() {
  const colors = useColors();
  const {open: openSidebar, navigate} = useSidebar();
  const {health, models, currentSoul, refresh: refreshModels} = useModelStore();
  const {sessions, refreshSessions, loadSession} = useChatStore();
  const [systemHealth, setSystemHealth] = useState<SystemHealth | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const fetchHealth = useCallback(async () => {
    try {
      const data = await api.get<SystemHealth>('/health');
      setSystemHealth(data);
    } catch {
      setSystemHealth(null);
    }
  }, []);

  const loadAll = useCallback(async () => {
    await Promise.all([refreshModels(), refreshSessions(), fetchHealth()]);
  }, [refreshModels, refreshSessions, fetchHealth]);

  useEffect(() => {
    loadAll().finally(() => setLoading(false));
  }, [loadAll]);

  const onRefresh = async () => {
    setRefreshing(true);
    await loadAll();
    setRefreshing(false);
  };

  const recentSessions = sessions.slice(0, 5);
  const isModelLoaded = health?.model_loaded ?? systemHealth?.model_loaded ?? false;
  const apiOk = systemHealth != null;

  return (
    <SafeAreaView style={{flex: 1, backgroundColor: 'var(--background)'}} edges={['top']}>
      <ScrollView
        style={{flex: 1}}
        contentContainerStyle={{padding: 16, gap: 12, paddingBottom: 32}}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}>

        {/* Header */}
        <XStack alignItems="center" gap={12} marginBottom={4}>
          <YStack
            width={40} height={40} borderRadius={14}
            alignItems="center" justifyContent="center"
            backgroundColor={colors.primaryAlpha(0.08)}
            onPress={openSidebar}
            pressStyle={{opacity: 0.6, scale: 0.95}}
            accessible accessibilityRole="button" accessibilityLabel="Open menu">
            <Icon name="menu" size={20} color={colors.primary} />
          </YStack>
          <YStack flex={1}>
            <Text fontSize={22} fontWeight="700" letterSpacing={-0.5} color="$color">Home</Text>
            <Text fontSize={12} color="$color10">SloughGPT Mobile</Text>
          </YStack>
          <Pressable onPress={() => navigate('Settings/Health')}>
            <YStack
              paddingHorizontal={10}
              paddingVertical={5}
              borderRadius={10}
              backgroundColor={apiOk ? colors.successAlpha(0.1) : colors.errorAlpha(0.1)}
              borderWidth={1}
              borderColor={apiOk ? colors.successAlpha(0.2) : colors.errorAlpha(0.2)}
              flexDirection="row"
              alignItems="center"
              gap={6}>
              <YStack width={6} height={6} borderRadius={3} backgroundColor={apiOk ? colors.success : colors.error} />
              <Text fontSize={11} fontWeight="600" color={apiOk ? colors.success : colors.error}>
                {apiOk ? 'Online' : 'Offline'}
              </Text>
            </YStack>
          </Pressable>
        </XStack>

        {/* Status Row */}
        <XStack gap={8}>
          <Card flex={1} padding={14} gap={8}>
            <XStack justifyContent="space-between" alignItems="center">
              <Text fontSize={11} color="$color10">Model</Text>
              <StatusBadge
                label={isModelLoaded ? 'Loaded' : 'None'}
                variant={isModelLoaded ? 'success' : 'default'}
              />
            </XStack>
            <Text fontSize={13} fontWeight="600" color="$color" numberOfLines={1}>
              {health?.model_name || systemHealth?.model_name || 'No model loaded'}
            </Text>
          </Card>

          <Card flex={1} padding={14} gap={8}>
            <Text fontSize={11} color="$color10">Soul</Text>
            <Text fontSize={13} fontWeight="600" color="$color" numberOfLines={1}>
              {currentSoul?.name || 'Default'}
            </Text>
            {currentSoul?.description && (
              <Text fontSize={11} color="$color10" numberOfLines={1}>{currentSoul.description}</Text>
            )}
          </Card>
        </XStack>

        {/* Quick Actions */}
        <Card>
          <SectionHeader icon="zap" title="Quick Actions" />
          <XStack gap={8}>
            <QuickAction icon="message-circle" label="Chat" onPress={() => navigate('Chat')} />
            <QuickAction icon="brain" label="Models" onPress={() => navigate('Models')} />
            <QuickAction icon="dumbbell" label="Train" onPress={() => navigate('Tools/Training')} />
            <QuickAction icon="book-open" label="Datasets" onPress={() => navigate('Tools/Datasets')} />
          </XStack>
        </Card>

        {/* System Stats */}
        {systemHealth && (
          <Card>
            <SectionHeader icon="heart-pulse" title="System" />
            <XStack gap={16}>
              <YStack gap={4}>
                <Text fontSize={10} color="$color10">Uptime</Text>
                <Text fontSize={14} fontWeight="600" color="$color">{formatUptime(systemHealth.uptime_s)}</Text>
              </YStack>
              <YStack gap={4}>
                <Text fontSize={10} color="$color10">Requests</Text>
                <Text fontSize={14} fontWeight="600" color="$color">{systemHealth.request_count.toLocaleString()}</Text>
              </YStack>
              <YStack gap={4}>
                <Text fontSize={10} color="$color10">Errors</Text>
                <Text fontSize={14} fontWeight="600" color={systemHealth.error_count > 0 ? colors.error : '$color'}>
                  {systemHealth.error_count}
                </Text>
              </YStack>
              <YStack gap={4}>
                <Text fontSize={10} color="$color10">Models</Text>
                <Text fontSize={14} fontWeight="600" color="$color">{models.length}</Text>
              </YStack>
            </XStack>
          </Card>
        )}

        {/* Recent Sessions */}
        <Card>
          <XStack justifyContent="space-between" alignItems="center">
            <SectionHeader icon="message-square" title="Recent Chats" />
            <Pressable onPress={() => navigate('Chat')}>
              <Text fontSize={12} fontWeight="600" color={colors.primary}>View all</Text>
            </Pressable>
          </XStack>
          {recentSessions.length === 0 ? (
            <YStack padding={24} alignItems="center" gap={8}>
              <YStack width={48} height={48} borderRadius={14} backgroundColor={colors.primaryAlpha(0.08)} alignItems="center" justifyContent="center">
                <Icon name="message-circle" size={24} color={colors.primary} />
              </YStack>
              <Text fontSize={14} fontWeight="600" color="$color">No conversations yet</Text>
              <Text fontSize={12} color="$color10">Start a chat to begin</Text>
            </YStack>
          ) : (
            recentSessions.map((session: Session) => (
              <Pressable
                key={session.id}
                onPress={() => {
                  loadSession(session.id);
                  navigate('Chat');
                }}>
                {({pressed}) => (
                  <XStack
                    padding={12}
                    borderRadius={12}
                    backgroundColor={pressed ? colors.primaryAlpha(0.06) : 'transparent'}
                    gap={12}
                    alignItems="center"
                    pressStyle={{opacity: 0.8}}>
                    <YStack width={36} height={36} borderRadius={10} backgroundColor={colors.primaryAlpha(0.1)} alignItems="center" justifyContent="center">
                      <Icon name="message-circle" size={16} color={colors.primary} />
                    </YStack>
                    <YStack flex={1} gap={2}>
                      <Text fontSize={13} fontWeight="600" color="$color" numberOfLines={1}>
                        {session.name || 'New chat'}
                      </Text>
                      <Text fontSize={11} color="$color10" numberOfLines={1}>
                        {session.message_count ?? 0} messages
                      </Text>
                    </YStack>
                    <Icon name="chevron-down" size={14} color={colors.textMuted} />
                  </XStack>
                )}
              </Pressable>
            ))
          )}
        </Card>

      </ScrollView>
    </SafeAreaView>
  );
}
