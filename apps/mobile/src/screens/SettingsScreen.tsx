import React, {useEffect, useState, useRef} from 'react';
import {
  ScrollView,
  TextInput as RNTextInput,
  Alert,
  Switch,
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {useNavigation} from '@react-navigation/native';
import {YStack, XStack, Text, Input, Button} from 'tamagui';
import {useColors} from '../theme/colors';
import {useSidebar} from '../contexts/SidebarContext';
import {toast} from '../services/toast';
import {Icon} from '../components/Icon';
import {useSettingsStore} from '../stores/settings-store';
import {useModelStore} from '../stores/model-store';
import {StatusBadge} from '../components/StatusBadge';
import {useHybridStore} from '../stores/hybrid-inference-store';
import {useProvidersStore} from '../stores/providers-store';
import {api, getApiUrl, setApiUrl} from '../services/api-client';
import {APP_VERSION} from '../constants';
import {
  registerForPushNotifications,
  unregisterPushNotifications,
  isNotificationsEnabled,
  onNotification,
} from '../services/push-notifications';
import {sounds} from '../services/sounds';
import {SettingsCard, SettingsCardHeader, SettingsSelectableChip} from '../components/SettingsCard';
import {triggerHaptic} from '../services/haptics';
import type {HealthStatus} from '../types';
import type {ThemeMode} from '../types';
import type {AccentColor} from '../stores/settings-store';
import type {FontFamilyOption, FontSizeScale} from '../stores/settings-store';

const FONT_FAMILY_OPTIONS: {label: string; value: FontFamilyOption}[] = [
  {label: 'System Default', value: 'system'},
  {label: 'Outfit', value: 'outfit'},
];

const FONT_SCALE_PRESETS: {label: string; value: FontSizeScale}[] = [
  {label: 'XS', value: 0.85},
  {label: 'SM', value: 0.925},
  {label: 'MD', value: 1.0},
  {label: 'LG', value: 1.1},
  {label: 'XL', value: 1.2},
];

export function SettingsScreen() {
  const colors = useColors();
  const settings = useSettingsStore();
  const {health, refresh} = useModelStore();
  const navigation = useNavigation<any>();
  const {open: openSidebar} = useSidebar();
  const [serverUrl, setServerUrl] = useState('');
  const serverUrlEditedRef = useRef(false);
  const [healthData, setHealthData] = useState<HealthStatus | null>(null);
  const [notificationsOn, setNotificationsOn] = useState(false);
  const [lastNotification, setLastNotification] = useState<string | null>(null);
  const [soundsOn, setSoundsOn] = useState(sounds.isEnabled());
  const hybrid = useHybridStore();
  const activeProviderId = useProvidersStore(s => s.activeProviderId);
  const providers = useProvidersStore(s => s.providers);

  useEffect(() => {
    getApiUrl().then(url => {
      if (!serverUrlEditedRef.current) {
        setServerUrl(url);
      }
    });
    const fetchHealth = () => api.get<HealthStatus>('/health').then(setHealthData).catch(() => {});
    fetchHealth();
    const healthTimer = setInterval(fetchHealth, 30000);
    isNotificationsEnabled().then(setNotificationsOn);
    return () => clearInterval(healthTimer);
  }, []);

  useEffect(() => {
    return onNotification((title, body) => {
      setLastNotification(`${title}: ${body}`);
    });
  }, []);

  const handleToggleNotifications = async (val: boolean) => {
    triggerHaptic('selection');
    if (val) {
      const token = await registerForPushNotifications();
      setNotificationsOn(!!token);
      if (token) {
        Alert.alert('Notifications enabled', 'You\'ll receive training and chat updates.');
      } else {
        Alert.alert('Permission denied', 'Enable notifications in system settings.');
      }
    } else {
      await unregisterPushNotifications();
      setNotificationsOn(false);
    }
  };

  const handleSaveUrl = async () => {
    const trimmed = serverUrl.trim();
    if (!trimmed) return;
    await setApiUrl(trimmed);
    await refresh();
    api.get<HealthStatus>('/health').then(setHealthData).catch(() => {});
  };

  const themes: ThemeMode[] = ['light', 'dark', 'system'];

  return (
    <SafeAreaView style={{flex: 1, backgroundColor: 'var(--background)'}} edges={['top']}>
      <KeyboardAvoidingView style={{flex: 1}} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
        <ScrollView contentContainerStyle={{padding: 16, paddingBottom: 40}}>
          {/* Header */}
          <XStack alignItems="center" gap={12} marginBottom={20}>
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
              <Text fontSize={22} fontWeight="700" letterSpacing={-0.5} color="$color">
                Settings
              </Text>
            </YStack>
          </XStack>

          {/* Server */}
          <SettingsCard>
            <SettingsCardHeader icon="terminal" title="Server" />
            <XStack justifyContent="space-between" alignItems="center">
              <Text fontSize={13} color="$color11">Status</Text>
              <StatusBadge
                label={healthData?.status === 'healthy' ? 'Connected' : 'Offline'}
                variant={healthData?.status === 'healthy' ? 'success' : 'error'}
              />
            </XStack>
            <XStack justifyContent="space-between" alignItems="center">
              <Text fontSize={13} color="$color11">Model</Text>
              <Text fontSize={13} fontWeight="500" color="$color">{healthData?.model_name || 'None'}</Text>
            </XStack>
            <XStack gap={8} marginTop={4}>
              <Input
                flex={1}
                size="$3"
                value={serverUrl}
                onChangeText={val => { serverUrlEditedRef.current = true; setServerUrl(val); }}
                placeholder="http://localhost:8000"
                autoCapitalize="none"
                autoCorrect={false}
                borderRadius={10}
                backgroundColor={colors.backgroundHover}
                borderWidth={1}
                borderColor={colors.border}
              />
              <Button
                size="$3"
                backgroundColor={colors.primary}
                color="white"
                fontWeight="600"
                borderRadius={10}
                pressStyle={{opacity: 0.8, scale: 0.97}}
                onPress={() => { triggerHaptic('success'); handleSaveUrl(); }}>
                Save
              </Button>
            </XStack>
          </SettingsCard>

          {/* System Health nav */}
          <YStack
            backgroundColor="$background"
            borderRadius={16}
            borderWidth={1}
            borderColor={colors.border}
            padding={16}
            marginBottom={4}
            shadowColor="black"
            shadowOffset={{width: 0, height: 2}}
            shadowOpacity={0.06}
            shadowRadius={8}
            elevation={2}
            onPress={() => { triggerHaptic('selection'); navigation.navigate('Health'); }}
            pressStyle={{opacity: 0.7, scale: 0.98}}>
            <XStack justifyContent="space-between" alignItems="center">
              <XStack alignItems="center" gap={10}>
                <YStack width={32} height={32} borderRadius={10} backgroundColor={colors.primaryAlpha(0.1)} alignItems="center" justifyContent="center">
                  <Icon name="heart-pulse" size={16} color={colors.primary} />
                </YStack>
                <YStack>
                  <Text fontSize={15} fontWeight="600" color="$color">System Health</Text>
                  <Text fontSize={12} color="$color11">CPU, memory, disk, uptime</Text>
                </YStack>
              </XStack>
              <Icon name="chevron-down" size={16} color={colors.textMuted} />
            </XStack>
          </YStack>

          {/* Inference */}
          <SettingsCard>
            <SettingsCardHeader icon="zap" title="Inference" />

            {/* Engine chips */}
            <XStack gap={8}>
              {(['slonet', 'qwen', 'remote'] as const).map(engine => {
                const active = hybrid.activeEngine === engine;
                const label = engine === 'slonet' ? 'SloNet' : engine === 'qwen' ? 'Qwen' : 'Server';
                return (
                  <YStack
                    key={engine}
                    flex={1}
                    paddingVertical={10}
                    borderRadius={12}
                    backgroundColor={active ? colors.primary : colors.backgroundHover}
                    borderWidth={1}
                    borderColor={active ? colors.primary : colors.border}
                    alignItems="center"
                    pressStyle={{opacity: 0.8, scale: 0.97}}
                    onPress={() => { triggerHaptic('selection'); hybrid.setActiveEngine(engine); }}>
                    <Text fontSize={12} fontWeight="600" color={active ? 'white' : '$color11'}>{label}</Text>
                  </YStack>
                );
              })}
            </XStack>

            {/* Provider selector (when remote is active) */}
            {hybrid.activeEngine === 'remote' && (
              <YStack gap={6}>
                <Text fontSize={11} fontWeight="600" color="$color11" textTransform="uppercase" letterSpacing={0.5}>
                  Remote Provider
                </Text>
                <XStack gap={6} flexWrap="wrap">
                  {[
                    {id: 'remote' as const, label: 'Self-Hosted'},
                    ...Object.entries(providers)
                      .filter(([_, p]) => p.apiKey && p.enabled)
                      .map(([id, p]) => ({id: id as any, label: p.name})),
                  ].map(({id, label}) => {
                    const active = id === 'remote'
                      ? !activeProviderId
                      : activeProviderId === id;
                    return (
                      <YStack
                        key={id}
                        paddingVertical={6}
                        paddingHorizontal={12}
                        borderRadius={10}
                        backgroundColor={active ? colors.primary : colors.backgroundHover}
                        borderWidth={1}
                        borderColor={active ? colors.primary : colors.border}
                        pressStyle={{opacity: 0.8}}
                        onPress={() => {
                          triggerHaptic('selection');
                          if (id === 'remote') {
                            useProvidersStore.getState().setActiveProvider(null);
                            hybrid.setActiveEngine('remote');
                          } else {
                            useProvidersStore.getState().setActiveProvider(id);
                            hybrid.setActiveEngine(id);
                          }
                        }}>
                        <Text fontSize={11} fontWeight="600" color={active ? 'white' : '$color11'}>
                          {label}
                        </Text>
                      </YStack>
                    );
                  })}
                  <YStack
                    paddingVertical={6}
                    paddingHorizontal={12}
                    borderRadius={10}
                    backgroundColor="transparent"
                    borderWidth={1}
                    borderColor={colors.border}
                    borderStyle="dashed"
                    pressStyle={{opacity: 0.7}}
                    onPress={() => {
                      triggerHaptic('light');
                      navigation.navigate('Providers');
                    }}>
                    <Text fontSize={11} fontWeight="500" color="$color11">+ Add</Text>
                  </YStack>
                </XStack>
              </YStack>
            )}

            {/* SloNet */}
            <YStack
              padding={12} borderRadius={12}
              backgroundColor={colors.backgroundHover}
              borderWidth={1}
              borderColor={colors.border}
              gap={8}>
              <XStack justifyContent="space-between" alignItems="center">
                <YStack flex={1}>
                  <Text fontSize={14} fontWeight="600" color="$color">SloNet</Text>
                  <Text fontSize={12} color="$color11">
                    {hybrid.slonet.loaded ? `Loaded — ${hybrid.slonet.modelName}` : 'Not loaded'}
                  </Text>
                </YStack>
                {hybrid.slonet.loaded ? (
                  <Button size="$2" backgroundColor={colors.background} color="$color11" fontWeight="600" borderRadius={10} borderWidth={1} borderColor={colors.border} pressStyle={{opacity: 0.7}} onPress={() => { triggerHaptic('medium'); hybrid.unloadSloNet(); }}>Unload</Button>
                ) : hybrid.slonet.downloadProgress !== null && hybrid.slonet.downloadProgress < 1 ? (
                  <ActivityIndicator size="small" color={colors.primary} />
                ) : (
                  <XStack gap={6}>
                    <Button size="$2" backgroundColor={colors.primary} color="white" fontWeight="600" borderRadius={10} pressStyle={{opacity: 0.8}} onPress={() => { triggerHaptic('medium'); hybrid.loadSloNet(); }}>Load</Button>
                    <Button size="$2" backgroundColor={colors.background} color="$color11" fontWeight="600" borderRadius={10} borderWidth={1} borderColor={colors.border} pressStyle={{opacity: 0.7}} onPress={() => { triggerHaptic('medium'); hybrid.loadSloNetFromSou(); }}>.sou</Button>
                  </XStack>
                )}
              </XStack>
            </YStack>

            {/* Qwen */}
            <YStack
              padding={12} borderRadius={12}
              backgroundColor={colors.backgroundHover}
              borderWidth={1}
              borderColor={colors.border}
              gap={8}>
              <XStack justifyContent="space-between" alignItems="center">
                <YStack flex={1}>
                  <Text fontSize={14} fontWeight="600" color="$color">Qwen 0.5B GGUF</Text>
                  <Text fontSize={12} color="$color11">
                    {hybrid.qwen.loaded
                      ? 'Loaded — 15-30 tok/s'
                      : hybrid.qwen.downloadProgress !== null && hybrid.qwen.downloadProgress < 1
                      ? `Downloading ${Math.round(hybrid.qwen.downloadProgress * 100)}%`
                      : 'Not downloaded'}
                  </Text>
                </YStack>
                {hybrid.qwen.loaded ? (
                  <Button size="$2" backgroundColor={colors.background} color="$color11" fontWeight="600" borderRadius={10} borderWidth={1} borderColor={colors.border} pressStyle={{opacity: 0.7}} onPress={() => { triggerHaptic('medium'); hybrid.unloadQwen(); }}>Unload</Button>
                ) : hybrid.qwen.downloadProgress !== null && hybrid.qwen.downloadProgress < 1 ? (
                  <YStack width={60} height={4} borderRadius={2} backgroundColor={colors.border} overflow="hidden">
                    <YStack height="100%" backgroundColor={colors.primary} borderRadius={2} width={`${hybrid.qwen.downloadProgress * 100}%`} />
                  </YStack>
                ) : (
                  <Button size="$2" backgroundColor={colors.primary} color="white" fontWeight="600" borderRadius={10} pressStyle={{opacity: 0.8}} onPress={() => { triggerHaptic('medium'); hybrid.loadQwen(); }}>Download</Button>
                )}
              </XStack>
            </YStack>

            {hybrid.lastError && (
              <Text fontSize={11} color="$color10">{hybrid.lastError}</Text>
            )}

            {/* Offline toggle */}
            <XStack
              gap={12} marginTop={4}
              padding={12} borderRadius={12}
              backgroundColor={hybrid.offlineOnly ? colors.primaryAlpha(0.08) : colors.backgroundHover}
              borderWidth={1}
              borderColor={hybrid.offlineOnly ? colors.primary : colors.border}
              alignItems="center"
              onPress={() => { triggerHaptic('selection'); hybrid.setOfflineOnly(!hybrid.offlineOnly); }}
              pressStyle={{opacity: 0.8}}>
              <YStack width={28} height={28} borderRadius={8} backgroundColor={hybrid.offlineOnly ? colors.primary : colors.border} alignItems="center" justifyContent="center">
                <Icon name={hybrid.offlineOnly ? 'check' : 'x'} size={14} color="white" />
              </YStack>
              <YStack flex={1}>
                <Text fontSize={14} fontWeight="600" color="$color">Offline Mode</Text>
                <Text fontSize={12} color="$color11">
                  {hybrid.offlineOnly
                    ? 'Server disabled — local engines only'
                    : 'Load SloNet or Qwen to enable'}
                </Text>
              </YStack>
              <YStack
                width={44} height={26} borderRadius={13}
                backgroundColor={hybrid.offlineOnly ? colors.primary : colors.border}
                alignItems={hybrid.offlineOnly ? 'flex-end' : 'flex-start'}
                justifyContent="center"
                paddingHorizontal={2}>
                <YStack width={22} height={22} borderRadius={11} backgroundColor="white" />
              </YStack>
            </XStack>
          </SettingsCard>

          {/* Appearance */}
          <YStack
            backgroundColor="$background"
            borderRadius={16}
            borderWidth={1}
            borderColor={colors.border}
            padding={16} gap={12}
            marginBottom={4}
            shadowColor="black"
            shadowOffset={{width: 0, height: 2}}
            shadowOpacity={0.06}
            shadowRadius={8}
            elevation={2}>
            <XStack alignItems="center" gap={10}>
              <YStack width={32} height={32} borderRadius={10} backgroundColor={colors.primaryAlpha(0.1)} alignItems="center" justifyContent="center">
                <Icon name="palette" size={16} color={colors.primary} />
              </YStack>
              <Text fontSize={15} fontWeight="600" color="$color">Appearance</Text>
            </XStack>
            <XStack gap={8}>
              {themes.map(t => {
                const active = settings.theme === t;
                return (
                  <YStack
                    key={t}
                    flex={1}
                    paddingVertical={10}
                    borderRadius={12}
                    backgroundColor={active ? colors.primary : colors.backgroundHover}
                    borderWidth={1}
                    borderColor={active ? colors.primary : colors.border}
                    alignItems="center"
                    pressStyle={{opacity: 0.8, scale: 0.97}}
                    onPress={() => { triggerHaptic('selection'); settings.setTheme(t); }}>
                    <Text fontSize={12} fontWeight="600" color={active ? 'white' : '$color11'}>
                      {t.charAt(0).toUpperCase() + t.slice(1)}
                    </Text>
                  </YStack>
                );
              })}
            </XStack>
            <XStack gap={10} flexWrap="wrap">
              {([
                {key: 'violet', color: '#7C52C4', label: 'Violet'},
                {key: 'rose', color: '#E11D48', label: 'Rose'},
                {key: 'amber', color: '#D97706', label: 'Amber'},
                {key: 'emerald', color: '#059669', label: 'Emerald'},
                {key: 'sky', color: '#0284C7', label: 'Sky'},
              ] as const).map(opt => {
                const active = settings.accentColor === opt.key;
                return (
                  <YStack alignItems="center" gap={4}>
                    <YStack
                      width={36}
                      height={36}
                      borderRadius={12}
                      backgroundColor={opt.color}
                      borderWidth={2}
                      borderColor={active ? 'white' : 'transparent'}
                      alignItems="center"
                      justifyContent="center"
                      shadowColor={active ? opt.color : 'transparent'}
                      shadowOffset={{width: 0, height: 4}}
                      shadowOpacity={active ? 0.4 : 0}
                      shadowRadius={8}
                      elevation={active ? 4 : 0}
                      pressStyle={{scale: 0.9}}
                      onPress={() => { triggerHaptic('selection'); settings.update({accentColor: opt.key}); }}>
                      {active && (
                        <Icon name="check" size={14} color="white" />
                      )}
                    </YStack>
                    <Text fontSize={10} color={active ? colors.primary : '$color10'} fontWeight={active ? '600' : '400'}>
                      {opt.label}
                    </Text>
                  </YStack>
                );
              })}
            </XStack>
          </YStack>

          {/* Font */}
          <YStack
            backgroundColor="$background"
            borderRadius={16}
            borderWidth={1}
            borderColor={colors.border}
            padding={16} gap={12}
            marginBottom={4}
            shadowColor="black"
            shadowOffset={{width: 0, height: 2}}
            shadowOpacity={0.06}
            shadowRadius={8}
            elevation={2}>
            <XStack alignItems="center" gap={10}>
              <YStack width={32} height={32} borderRadius={10} backgroundColor={colors.primaryAlpha(0.1)} alignItems="center" justifyContent="center">
                <Icon name="book-open" size={16} color={colors.primary} />
              </YStack>
              <Text fontSize={15} fontWeight="600" color="$color">Font</Text>
            </XStack>
            <Text fontSize={12} color="$color11">Typeface</Text>
            <XStack gap={8}>
              {FONT_FAMILY_OPTIONS.map(opt => {
                const active = settings.fontFamily === opt.value;
                return (
                  <YStack
                    key={opt.value}
                    flex={1}
                    paddingVertical={10}
                    borderRadius={12}
                    backgroundColor={active ? colors.primary : colors.backgroundHover}
                    borderWidth={1}
                    borderColor={active ? colors.primary : colors.border}
                    alignItems="center"
                    pressStyle={{opacity: 0.8, scale: 0.97}}
                    onPress={() => { triggerHaptic('selection'); settings.setFontFamily(opt.value); }}>
                    <Text fontSize={12} fontWeight="600" color={active ? 'white' : '$color11'}>{opt.label}</Text>
                  </YStack>
                );
              })}
            </XStack>
            <Text fontSize={12} color="$color11" marginTop={2}>Size</Text>
            <XStack gap={6} flexWrap="wrap">
              {FONT_SCALE_PRESETS.map(p => {
                const active = settings.fontSizeScale === p.value;
                return (
                  <YStack
                    key={p.value}
                    paddingHorizontal={14} paddingVertical={8}
                    borderRadius={10}
                    backgroundColor={active ? colors.primary : colors.backgroundHover}
                    borderWidth={1}
                    borderColor={active ? colors.primary : colors.border}
                    pressStyle={{opacity: 0.8, scale: 0.97}}
                    onPress={() => { triggerHaptic('selection'); settings.setFontSizeScale(p.value); }}>
                    <Text fontSize={12} fontWeight="600" color={active ? 'white' : '$color11'}>{p.label}</Text>
                  </YStack>
                );
              })}
            </XStack>
          </YStack>

          {/* Push Notifications */}
          <YStack
            backgroundColor="$background"
            borderRadius={16}
            borderWidth={1}
            borderColor={colors.border}
            padding={16}
            marginBottom={4}
            shadowColor="black"
            shadowOffset={{width: 0, height: 2}}
            shadowOpacity={0.06}
            shadowRadius={8}
            elevation={2}>
            <XStack justifyContent="space-between" alignItems="center">
              <XStack alignItems="center" gap={10} flex={1}>
                <YStack width={32} height={32} borderRadius={10} backgroundColor={colors.primaryAlpha(0.1)} alignItems="center" justifyContent="center">
                  <Icon name="message-square" size={16} color={colors.primary} />
                </YStack>
                <YStack flex={1}>
                  <Text fontSize={15} fontWeight="600" color="$color">Push Notifications</Text>
                  <Text fontSize={12} color="$color11">Training and chat updates</Text>
                </YStack>
              </XStack>
              <Switch
                value={notificationsOn}
                onValueChange={handleToggleNotifications}
                trackColor={{false: colors.border, true: colors.primary + '60'}}
                thumbColor={notificationsOn ? colors.primary : colors.textMuted}
              />
            </XStack>
            {lastNotification && (
              <YStack
                backgroundColor={colors.backgroundHover}
                marginTop={8} borderRadius={10} padding={10}
                borderWidth={1} borderColor={colors.border}>
                <Text fontSize={11} color="$color11">{lastNotification}</Text>
              </YStack>
            )}
          </YStack>

          {/* Sound Effects */}
          <YStack
            backgroundColor="$background"
            borderRadius={16}
            borderWidth={1}
            borderColor={colors.border}
            padding={16}
            marginBottom={4}
            shadowColor="black"
            shadowOffset={{width: 0, height: 2}}
            shadowOpacity={0.06}
            shadowRadius={8}
            elevation={2}>
            <XStack justifyContent="space-between" alignItems="center">
              <XStack alignItems="center" gap={10} flex={1}>
                <YStack width={32} height={32} borderRadius={10} backgroundColor={colors.primaryAlpha(0.1)} alignItems="center" justifyContent="center">
                  <Icon name="music" size={16} color={colors.primary} />
                </YStack>
                <YStack flex={1}>
                  <Text fontSize={15} fontWeight="600" color="$color">Sound Effects</Text>
                  <Text fontSize={12} color="$color11">Audio feedback on send/receive</Text>
                </YStack>
              </XStack>
              <Switch
                value={soundsOn}
                onValueChange={(val) => { triggerHaptic('selection'); setSoundsOn(val); sounds.setEnabled(val); }}
                trackColor={{false: colors.border, true: colors.primary + '60'}}
                thumbColor={soundsOn ? colors.primary : colors.textMuted}
              />
            </XStack>
          </YStack>

          {/* Chat Defaults */}
          <YStack
            backgroundColor="$background"
            borderRadius={16}
            borderWidth={1}
            borderColor={colors.border}
            padding={16} gap={12}
            marginBottom={4}
            shadowColor="black"
            shadowOffset={{width: 0, height: 2}}
            shadowOpacity={0.06}
            shadowRadius={8}
            elevation={2}>
            <XStack alignItems="center" gap={10}>
              <YStack width={32} height={32} borderRadius={10} backgroundColor={colors.primaryAlpha(0.1)} alignItems="center" justifyContent="center">
                <Icon name="settings" size={16} color={colors.primary} />
              </YStack>
              <Text fontSize={15} fontWeight="600" color="$color">Chat Defaults</Text>
            </XStack>
            {[
              {label: 'Temperature', value: settings.temperature.toFixed(1), key: 'temperature', options: [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]},
              {label: 'Max Tokens', value: String(settings.maxTokens), key: 'maxTokens', options: [128, 256, 512, 1024], exact: true},
              {label: 'Top-P', value: settings.topP.toFixed(1), key: 'topP', options: [0.7, 0.8, 0.9, 1.0]},
              {label: 'Top-K', value: String(settings.topK), key: 'topK', options: [20, 50, 100, 200], exact: true},
              {label: 'Repetition Penalty', value: settings.repetitionPenalty.toFixed(1), key: 'repetitionPenalty', options: [1.0, 1.1, 1.2, 1.5, 2.0]},
            ].map(({label, value, key, options, exact}) => (
              <YStack key={key} gap={6}>
                <XStack justifyContent="space-between" alignItems="center">
                  <Text fontSize={13} color="$color11">{label}</Text>
                  <Text fontSize={13} fontWeight="600" color={colors.primary}>{value}</Text>
                </XStack>
                <XStack gap={4} flexWrap="wrap">
                  {options.map(v => {
                    const match = exact
                      ? (settings as any)[key] === v
                      : Math.abs((settings as any)[key] - v) < 0.05;
                    return (
                      <YStack
                        key={String(v)}
                        paddingHorizontal={12} paddingVertical={6}
                        borderRadius={8}
                        backgroundColor={match ? colors.primary : colors.backgroundHover}
                        borderWidth={1}
                        borderColor={match ? colors.primary : colors.border}
                        pressStyle={{opacity: 0.8, scale: 0.97}}
                        onPress={() => { triggerHaptic('selection'); settings.update({[key]: v}); }}>
                        <Text fontSize={11} fontWeight="600" color={match ? 'white' : '$color11'}>
                          {exact ? String(v) : v.toFixed(1)}
                        </Text>
                      </YStack>
                    );
                  })}
                </XStack>
              </YStack>
            ))}
          </YStack>

          {/* Chat Background */}
          <YStack
            backgroundColor="$background"
            borderRadius={16}
            borderWidth={1}
            borderColor={colors.border}
            padding={16} gap={12}
            marginBottom={4}
            shadowColor="black"
            shadowOffset={{width: 0, height: 2}}
            shadowOpacity={0.06}
            shadowRadius={8}
            elevation={2}>
            <XStack alignItems="center" gap={10}>
              <YStack width={32} height={32} borderRadius={10} backgroundColor={colors.primaryAlpha(0.1)} alignItems="center" justifyContent="center">
                <Icon name="image" size={16} color={colors.primary} />
              </YStack>
              <Text fontSize={15} fontWeight="600" color="$color">Chat Background</Text>
            </XStack>
            <Text fontSize={12} color="$color11">Customize the chat area tint</Text>
            <XStack gap={8} flexWrap="wrap">
              {([
                {label: 'Default', value: ''},
                {label: 'Warm', value: 'rgba(252, 248, 240, 0.6)'},
                {label: 'Cool', value: 'rgba(240, 242, 252, 0.6)'},
                {label: 'Violet', value: 'rgba(244, 240, 252, 0.6)'},
                {label: 'Mint', value: 'rgba(240, 250, 244, 0.6)'},
                {label: 'Peach', value: 'rgba(252, 244, 240, 0.6)'},
              ] as const).map(opt => {
                const active = settings.chatBackground === opt.value;
                return (
                  <YStack
                    key={opt.label}
                    paddingHorizontal={14} paddingVertical={8}
                    borderRadius={10}
                    backgroundColor={active ? colors.primary : colors.backgroundHover}
                    borderWidth={1}
                    borderColor={active ? colors.primary : colors.border}
                    pressStyle={{opacity: 0.8, scale: 0.97}}
                    onPress={() => { triggerHaptic('selection'); settings.update({chatBackground: opt.value}); }}>
                    <Text fontSize={12} fontWeight="600" color={active ? 'white' : '$color11'}>
                      {opt.label}
                    </Text>
                  </YStack>
                );
              })}
            </XStack>
          </YStack>

          {/* Memory Context */}
          <YStack
            backgroundColor="$background"
            borderRadius={16}
            borderWidth={1}
            borderColor={colors.border}
            padding={16} gap={12}
            marginBottom={4}
            shadowColor="black"
            shadowOffset={{width: 0, height: 2}}
            shadowOpacity={0.06}
            shadowRadius={8}
            elevation={2}>
            <XStack alignItems="center" gap={10}>
              <YStack width={32} height={32} borderRadius={10} backgroundColor={colors.primaryAlpha(0.1)} alignItems="center" justifyContent="center">
                <Icon name="brain" size={16} color={colors.primary} />
              </YStack>
              <Text fontSize={15} fontWeight="600" color="$color">Memory Context</Text>
            </XStack>
            <Text fontSize={12} color="$color11">Custom context the AI always remembers</Text>
            <RNTextInput
              style={{
                fontSize: 14, color: colors.text, backgroundColor: colors.backgroundHover,
                borderRadius: 12, padding: 14, minHeight: 80,
                borderWidth: 1, borderColor: colors.border, textAlignVertical: 'top',
              }}
              value={settings.memoryContext}
              onChangeText={v => settings.update({memoryContext: v})}
              placeholder="I prefer concise answers. My expertise is in..."
              placeholderTextColor={colors.textMuted}
              multiline
            />
          </YStack>

          {/* Nav cards */}
          {[
            {label: 'Bookmarks', desc: 'Saved messages for quick access', target: 'Bookmarks', icon: 'bookmark'},
            {label: 'Providers', desc: 'Connect OpenAI, Anthropic, Google, and more', target: 'Providers', icon: 'layers'},
            {label: 'About SloughGPT', desc: 'Version, features, architecture', target: 'About', icon: 'info'},
            {label: 'Training', desc: 'Fine-tune models by chatting and interacting', target: 'Training', icon: 'dumbbell'},
            {label: 'What AI Knows About Me', desc: "View and manage the AI's knowledge about you", target: 'Knowledge', icon: 'search'},
            {label: 'Help', desc: 'FAQ, keyboard shortcuts, troubleshooting', target: 'Help', icon: 'help'},
            {label: 'Search Messages', desc: 'Find messages across conversations', target: 'Search', icon: 'search'},
          ].map(({label, desc, target, icon}) => (
            <YStack
              key={target}
              backgroundColor="$background"
              borderRadius={16}
              borderWidth={1}
              borderColor={colors.border}
              padding={16}
              marginBottom={4}
              shadowColor="black"
              shadowOffset={{width: 0, height: 2}}
              shadowOpacity={0.06}
              shadowRadius={8}
              elevation={2}
              pressStyle={{opacity: 0.7, scale: 0.98}}
              onPress={() => { triggerHaptic('selection'); navigation.navigate(target); }}>
              <XStack justifyContent="space-between" alignItems="center">
                <XStack alignItems="center" gap={10}>
                  <YStack width={32} height={32} borderRadius={10} backgroundColor={colors.primaryAlpha(0.1)} alignItems="center" justifyContent="center">
                    <Icon name={icon as any} size={16} color={colors.primary} />
                  </YStack>
                  <YStack>
                    <Text fontSize={15} fontWeight="600" color="$color">{label}</Text>
                    <Text fontSize={12} color="$color11">{desc}</Text>
                  </YStack>
                </XStack>
                <Icon name="chevron-down" size={16} color={colors.textMuted} />
              </XStack>
            </YStack>
          ))}

          {/* Danger Zone */}
          <YStack
            backgroundColor="$background"
            borderRadius={16}
            borderWidth={1}
            borderColor={colors.error + '30'}
            padding={16} gap={12}
            marginBottom={4}>
            <XStack alignItems="center" gap={10}>
              <YStack width={32} height={32} borderRadius={10} backgroundColor={colors.error + '15'} alignItems="center" justifyContent="center">
                <Icon name="trash-2" size={16} color={colors.error} />
              </YStack>
              <Text fontSize={15} fontWeight="600" color={colors.error}>Danger Zone</Text>
            </XStack>
            <YStack
              paddingVertical={12} paddingHorizontal={16} borderRadius={12}
              backgroundColor={colors.error + '08'}
              alignItems="center"
              borderWidth={1}
              borderColor={colors.error + '20'}
              pressStyle={{opacity: 0.7, scale: 0.98}}
              onPress={() => {
                triggerHaptic('medium');
                Alert.alert('Reset Settings', 'Reset all settings to defaults?', [
                  {text: 'Cancel', style: 'cancel'},
                  {text: 'Reset', style: 'destructive', onPress: settings.reset},
                ]);
              }}>
              <Text fontSize={13} fontWeight="600" color={colors.error}>Reset all settings</Text>
            </YStack>
          </YStack>

          <YStack alignItems="center" paddingVertical={32}>
            <Text fontSize={11} color="$color10" opacity={0.5}>SloughGPT v{APP_VERSION}</Text>
          </YStack>
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}
