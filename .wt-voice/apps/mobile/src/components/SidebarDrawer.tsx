import React, {useEffect, useRef} from 'react';
import {Animated, PanResponder, Pressable, Dimensions, BackHandler} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {YStack, XStack, Text} from 'tamagui';
import {useColors} from '../theme/colors';
import {S, R, T} from '../theme/scales';
import {useSidebar} from '../contexts/SidebarContext';
import {useChatStore} from '../stores/chat-store';
import {useModelStore} from '../stores/model-store';
import {Icon} from '../components/Icon';

const SCREEN_WIDTH = Dimensions.get('window').width;
const DRAWER_WIDTH = Math.min(300, SCREEN_WIDTH * 0.82);
const SWIPE_STRIP_WIDTH = 20;

interface NavItem {
  icon: string;
  label: string;
  screen: string;
  badge?: number;
}

const NAV_ITEMS: NavItem[] = [
  {icon: 'home', label: 'Home', screen: 'Home'},
  {icon: 'message-circle', label: 'Chat', screen: 'Chat'},
  {icon: 'brain', label: 'Models', screen: 'Models'},
  {icon: 'dumbbell', label: 'Tools', screen: 'Tools'},
  {icon: 'settings', label: 'Settings', screen: 'Settings'},
];

export function SidebarDrawer() {
  const colors = useColors();
  const {visible, open, close, activeScreen, navigate} = useSidebar();
  const openRef = useRef(open);
  openRef.current = open;
  const sessions = useChatStore(s => s.sessions);
  const loadSession = useChatStore(s => s.loadSession);
  const activeSessionId = useChatStore(s => s.activeSessionId);
  const health = useModelStore(s => s.health);
  const translateX = useRef(new Animated.Value(-DRAWER_WIDTH)).current;
  const overlayOpacity = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    if (visible) {
      Animated.parallel([
        Animated.spring(translateX, {toValue: 0, useNativeDriver: true, damping: 25, stiffness: 200}),
        Animated.timing(overlayOpacity, {toValue: 1, duration: 200, useNativeDriver: true}),
      ]).start();
    } else {
      Animated.parallel([
        Animated.spring(translateX, {toValue: -DRAWER_WIDTH, useNativeDriver: true, damping: 25, stiffness: 200}),
        Animated.timing(overlayOpacity, {toValue: 0, duration: 150, useNativeDriver: true}),
      ]).start();
    }
  }, [visible]);

  useEffect(() => {
    if (!visible) return;
    const sub = BackHandler.addEventListener('hardwareBackPress', () => {
      close();
      return true;
    });
    return () => sub.remove();
  }, [visible, close]);

  const handleNav = (screen: string) => {
    navigate(screen);
  };

  const recentSessions = sessions.slice(0, 8);

  return (
    <>
      {/* Swipe-open strip (left edge, only when closed) */}
      {!visible && (
        <Animated.View
          style={{
            position: 'absolute', top: 0, left: 0, bottom: 0,
            width: SWIPE_STRIP_WIDTH, zIndex: 997,
          }}
          {...PanResponder.create({
            onMoveShouldSetPanResponder: (_, g) => g.dx > 10 && Math.abs(g.dx) > Math.abs(g.dy),
            onPanResponderRelease: (_, g) => {
              if (g.dx > SWIPE_STRIP_WIDTH) {
                openRef.current();
              }
            },
          }).panHandlers}
        />
      )}

      {/* Full-screen touchable overlay + drawer */}
      {visible && (
        <Pressable
          style={{
            position: 'absolute', top: 0, left: 0, right: 0, bottom: 0,
            zIndex: 998,
          }}
          onPress={close}>
          {/* Dimmed background */}
          <Animated.View
            style={{
              position: 'absolute', top: 0, left: 0, right: 0, bottom: 0,
              backgroundColor: 'rgba(0,0,0,0.4)',
              opacity: overlayOpacity,
            }}
          />

          {/* Drawer panel — absorb touches so they don't close */}
          <Animated.View
            onStartShouldSetResponder={() => true}
            style={{
              position: 'absolute', top: 0, left: 0, bottom: 0,
              width: DRAWER_WIDTH,
              transform: [{translateX}],
              backgroundColor: colors.background,
              borderRightWidth: 0.5,
              borderRightColor: colors.border,
              zIndex: 999,
            }}>
            <SafeAreaView style={{flex: 1}} edges={['top', 'bottom']}>
          {/* Header */}
          <XStack
            paddingHorizontal={S.xl} paddingVertical={S.lg}
            borderBottomWidth={0.5} borderBottomColor={colors.border}
            alignItems="center" justifyContent="space-between">
            <YStack gap={2}>
              <Text {...T.page} fontWeight="700" letterSpacing={-0.5} color={colors.primary}>
                SloughGPT
              </Text>
              <Text {...T.caption} color={colors.textMuted}>Mobile</Text>
            </YStack>
            <Pressable
              onPress={close}
              hitSlop={12}
              style={({pressed}) => ({
                width: 40, height: 40, borderRadius: R.md,
                alignItems: 'center', justifyContent: 'center',
                opacity: pressed ? 0.5 : 1,
              })}>
              <Icon name="x" size={20} color={colors.textSecondary} />
            </Pressable>
          </XStack>

          {/* Status */}
          <XStack
            paddingHorizontal={S.xl} paddingVertical={S.md}
            alignItems="center" gap={S.sm}>
            <YStack
              width={7} height={7} borderRadius={R.full}
              backgroundColor={health?.model_loaded ? colors.success : colors.textMuted}
            />
            <Text {...T.caption} color={colors.textMuted}>
              {health?.model_loaded ? `Connected — ${health.model_name || 'Model loaded'}` : 'No model loaded'}
            </Text>
          </XStack>

          {/* Navigation */}
          <YStack paddingHorizontal={S.sm} paddingTop={S.xs}>
            {NAV_ITEMS.map(item => {
              const isActive = activeScreen === item.screen || activeScreen.startsWith(item.screen + '/');
              return (
                <Pressable
                  key={item.screen}
                  onPress={() => handleNav(item.screen)}
                  style={({pressed}) => ({
                    flexDirection: 'row', alignItems: 'center', gap: S.md,
                    paddingHorizontal: S.lg, paddingVertical: S.md,
                    marginHorizontal: S.sm, borderRadius: R.md,
                    backgroundColor: isActive
                      ? colors.primaryAlpha(0.1)
                      : pressed
                        ? colors.primaryAlpha(0.06)
                        : 'transparent',
                  })}>
                  <Icon name={item.icon as any} size={18} color={isActive ? colors.primary : colors.textSecondary} />
                  <Text
                    {...T.body} fontWeight={isActive ? '600' : '400'}
                    color={isActive ? colors.primary : colors.text}>
                    {item.label}
                  </Text>
                </Pressable>
              );
            })}
          </YStack>

          {/* Divider */}
          <YStack height={0.5} backgroundColor={colors.border} marginHorizontal={S.xl} marginVertical={S.sm} />

          {/* Recent Sessions */}
          <YStack flex={1}>
            <XStack paddingHorizontal={S.xl} paddingVertical={S.sm} alignItems="center" justifyContent="space-between">
              <Text {...T.label} fontWeight="600" color={colors.textMuted} textTransform="uppercase">
                Recent
              </Text>
              <Pressable
                onPress={() => navigate('Chat')}
                style={({pressed}) => ({
                  paddingHorizontal: S.sm, paddingVertical: S.xs, borderRadius: R.sm,
                  backgroundColor: colors.primaryAlpha(0.08),
                  opacity: pressed ? 0.6 : 1,
                })}>
                <Text {...T.badge} fontWeight="600" color={colors.primary}>New</Text>
              </Pressable>
            </XStack>

            {recentSessions.length === 0 ? (
              <YStack padding={S.xl} alignItems="center" gap={S.sm}>
                <Icon name="message-circle" size={20} color={colors.textMuted} />
                <Text {...T.caption} color={colors.textMuted}>No conversations yet</Text>
              </YStack>
            ) : (
              recentSessions.map(session => {
                const isActive = session.id === activeSessionId;
                return (
                  <Pressable
                    key={session.id}
                    onPress={() => { loadSession(session.id); navigate('Chat'); }}
                    style={({pressed}) => ({
                      flexDirection: 'row', alignItems: 'center', gap: S.md,
                      paddingHorizontal: S.xl, paddingVertical: S.md,
                      marginHorizontal: S.sm, marginVertical: 1, borderRadius: R.md,
                      backgroundColor: isActive
                        ? colors.primaryAlpha(0.08)
                        : pressed
                          ? colors.primaryAlpha(0.04)
                          : 'transparent',
                    })}>
                    <YStack width={28} height={28} borderRadius={R.md} backgroundColor={colors.primaryAlpha(0.08)} alignItems="center" justifyContent="center">
                      <Icon name="message-circle" size={12} color={colors.primary} />
                    </YStack>
                    <YStack flex={1} gap={1}>
                      <Text {...T.caption} fontWeight={isActive ? '600' : '400'} color={colors.text} numberOfLines={1}>
                        {session.name || 'New conversation'}
                      </Text>
                      <Text {...T.badge} color={colors.textMuted}>
                        {session.message_count ?? 0} messages
                      </Text>
                    </YStack>
                  </Pressable>
                );
              })
            )}
          </YStack>

          {/* Footer */}
          <Pressable
            onPress={() => handleNav('Settings')}
            style={({pressed}) => ({
              flexDirection: 'row', alignItems: 'center', gap: S.md,
              paddingHorizontal: S.xl, paddingVertical: S.md,
              borderTopWidth: 0.5, borderTopColor: colors.border,
              backgroundColor: pressed ? colors.primaryAlpha(0.04) : 'transparent',
            })}>
            <Icon name="settings" size={16} color={colors.textSecondary} />
            <Text {...T.caption} color={colors.textSecondary}>Settings</Text>
          </Pressable>
        </SafeAreaView>
          </Animated.View>
        </Pressable>
      )}
    </>
  );
}
