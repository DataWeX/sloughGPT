import React, {useEffect, useState, useCallback} from 'react';
import {Pressable, RefreshControl, ScrollView} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {useNavigation} from '@react-navigation/native';
import type {NativeStackNavigationProp} from '@react-navigation/native-stack';
import {YStack, XStack, Text} from 'tamagui';
import {useColors} from '../theme/colors';
import {useModelStore} from '../stores/model-store';
import {useChatStore} from '../stores/chat-store';
import {api} from '../services/api-client';
import {Icon} from '../components/Icon';
import {StatusBadge} from '../components/StatusBadge';
import {triggerHaptic} from '../services/haptics';
import type {ToolsStackParamList} from '../navigation/types';
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

function QuickAction({icon, label, onPress, color}: {icon: string; label: string; onPress: () => void; color: string}) {
  const colors = useColors();
  return (
    <Pressable onPress={onPress}>
      {({pressed}) => (
        <YStack
          width={72}
          alignItems="center"
          gap={6}
          padding={10}
          borderRadius={12}
          backgroundColor={pressed ? colors.primaryAlpha(0.08) : 'transparent'}>
          <YStack
            width={44}
            height={44}
            borderRadius={12}
            backgroundColor={color + '18'}
            alignItems="center"
            justifyContent="center">
            <Icon name={icon as any} size={20} color={color} />
          </YStack>
          <Text fontSize={11} fontWeight="500" color={colors.text} textAlign="center" numberOfLines={1}>
            {label}
          </Text>
        </YStack>
      )}
    </Pressable>
  );
}

export function HomeScreen() {
  const colors = useColors();
  const navigation = useNavigation<NativeStackNavigationProp<ToolsStackParamList>>();
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
  const modelOk = isModelLoaded;
  const apiOk = systemHealth != null;

  return (
    <SafeAreaView style={{flex: 1, backgroundColor: colors.background}} edges={['top']}>
      <ScrollView
        style={{flex: 1}}
        contentContainerStyle={{padding: 16, gap: 16, paddingBottom: 32}}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}>

        {/* Header */}
        <XStack justifyContent="space-between" alignItems="center">
          <YStack gap={2}>
            <Text fontSize={24} fontWeight="700" letterSpacing={-0.3} color={colors.text}>Home</Text>
            <Text fontSize={12} color={colors.textMuted}>SloughGPT Mobile</Text>
          </YStack>
          <Pressable onPress={() => navigation.getParent()?.navigate('Tools', {screen: 'Health'})}>
            <YStack
              paddingHorizontal={8}
              paddingVertical={4}
              borderRadius={8}
              backgroundColor={apiOk ? colors.successAlpha(0.1) : colors.errorAlpha(0.1)}
              gap={4}
              alignItems="center"
              flexDirection="row">
              <YStack width={6} height={6} borderRadius={3} backgroundColor={apiOk ? colors.success : colors.error} />
              <Text fontSize={11} fontWeight="500" color={apiOk ? colors.success : colors.error}>
                {apiOk ? 'Online' : 'Offline'}
              </Text>
            </YStack>
          </Pressable>
        </XStack>

        {/* Status Row */}
        <XStack gap={8}>
          <YStack flex={1} padding={12} borderRadius={10} backgroundColor={colors.white} borderWidth={0.5} borderColor={colors.border} gap={6}>
            <XStack justifyContent="space-between" alignItems="center">
              <Text fontSize={11} color={colors.textMuted}>Model</Text>
              <StatusBadge
                label={isModelLoaded ? 'Loaded' : 'None'}
                variant={isModelLoaded ? 'success' : 'default'}
              />
            </XStack>
            <Text fontSize={13} fontWeight="600" color={colors.text} numberOfLines={1}>
              {health?.model_name || systemHealth?.model_name || 'No model loaded'}
            </Text>
          </YStack>

          <YStack flex={1} padding={12} borderRadius={10} backgroundColor={colors.white} borderWidth={0.5} borderColor={colors.border} gap={6}>
            <Text fontSize={11} color={colors.textMuted}>Soul</Text>
            <Text fontSize={13} fontWeight="600" color={colors.text} numberOfLines={1}>
              {currentSoul?.name || 'Default'}
            </Text>
            {currentSoul?.description && (
              <Text fontSize={11} color={colors.textMuted} numberOfLines={1}>{currentSoul.description}</Text>
            )}
          </YStack>
        </XStack>

        {/* Quick Actions */}
        <YStack padding={14} borderRadius={12} backgroundColor={colors.white} borderWidth={0.5} borderColor={colors.border} gap={10}>
          <Text fontSize={14} fontWeight="600" color={colors.text}>Quick Actions</Text>
          <XStack justifyContent="space-between">
            <QuickAction
              icon="message-circle"
              label="Chat"
              color={colors.primary}
              onPress={() => navigation.getParent()?.navigate('Chat')}
            />
            <QuickAction
              icon="brain"
              label="Models"
              color={colors.primary}
              onPress={() => navigation.getParent()?.navigate('Models')}
            />
            <QuickAction
              icon="dumbbell"
              label="Train"
              color={colors.primary}
              onPress={() => navigation.getParent()?.navigate('Tools', {screen: 'Training'})}
            />
            <QuickAction
              icon="database"
              label="Datasets"
              color={colors.primary}
              onPress={() => navigation.getParent()?.navigate('Tools', {screen: 'Datasets'})}
            />
          </XStack>
        </YStack>

        {/* System Stats */}
        {systemHealth && (
          <YStack padding={14} borderRadius={12} backgroundColor={colors.white} borderWidth={0.5} borderColor={colors.border} gap={8}>
            <Text fontSize={14} fontWeight="600" color={colors.text}>System</Text>
            <XStack gap={16}>
              <YStack gap={2}>
                <Text fontSize={10} color={colors.textMuted}>Uptime</Text>
                <Text fontSize={13} fontWeight="500" color={colors.text}>{formatUptime(systemHealth.uptime_s)}</Text>
              </YStack>
              <YStack gap={2}>
                <Text fontSize={10} color={colors.textMuted}>Requests</Text>
                <Text fontSize={13} fontWeight="500" color={colors.text}>{systemHealth.request_count.toLocaleString()}</Text>
              </YStack>
              <YStack gap={2}>
                <Text fontSize={10} color={colors.textMuted}>Errors</Text>
                <Text fontSize={13} fontWeight="500" color={systemHealth.error_count > 0 ? colors.error : colors.text}>
                  {systemHealth.error_count}
                </Text>
              </YStack>
              <YStack gap={2}>
                <Text fontSize={10} color={colors.textMuted}>Models</Text>
                <Text fontSize={13} fontWeight="500" color={colors.text}>{models.length}</Text>
              </YStack>
            </XStack>
          </YStack>
        )}

        {/* Recent Sessions */}
        <YStack padding={14} borderRadius={12} backgroundColor={colors.white} borderWidth={0.5} borderColor={colors.border} gap={10}>
          <XStack justifyContent="space-between" alignItems="center">
            <Text fontSize={14} fontWeight="600" color={colors.text}>Recent Chats</Text>
            <Pressable onPress={() => navigation.getParent()?.navigate('Chat')}>
              <Text fontSize={12} color={colors.primary}>View all</Text>
            </Pressable>
          </XStack>
          {recentSessions.length === 0 ? (
            <YStack padding={16} alignItems="center" gap={6}>
              <Icon name="message-circle" size={24} color={colors.textMuted} />
              <Text fontSize={13} color={colors.textMuted}>No conversations yet</Text>
              <Text fontSize={11} color={colors.textMuted}>Start a chat to begin</Text>
            </YStack>
          ) : (
            recentSessions.map((session: Session) => (
              <Pressable
                key={session.id}
                onPress={() => {
                  loadSession(session.id);
                  navigation.getParent()?.navigate('Chat');
                }}>
                {({pressed}) => (
                  <XStack
                    padding={10}
                    borderRadius={8}
                    backgroundColor={pressed ? colors.primaryAlpha(0.04) : 'transparent'}
                    gap={10}
                    alignItems="center">
                    <YStack width={32} height={32} borderRadius={8} backgroundColor={colors.primaryAlpha(0.1)} alignItems="center" justifyContent="center">
                      <Icon name="message-circle" size={14} color={colors.primary} />
                    </YStack>
                    <YStack flex={1} gap={2}>
                      <Text fontSize={13} fontWeight="500" color={colors.text} numberOfLines={1}>
                        {session.name || 'New chat'}
                      </Text>
                      <Text fontSize={11} color={colors.textMuted} numberOfLines={1}>
                        {session.message_count ?? 0} messages
                      </Text>
                    </YStack>
                    <Icon name="chevron-down" size={14} color={colors.textMuted} />
                  </XStack>
                )}
              </Pressable>
            ))
          )}
        </YStack>

      </ScrollView>
    </SafeAreaView>
  );
}
