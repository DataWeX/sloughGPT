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
import {YStack, XStack, Text, Input, Button, useTheme} from 'tamagui';
import {useSettingsStore} from '../stores/settings-store';
import {useModelStore} from '../stores/model-store';
import {StatusBadge} from '../components/StatusBadge';
import {useHybridStore} from '../stores/hybrid-inference-store';
import {api, getApiUrl, setApiUrl} from '../services/api-client';
import {
  registerForPushNotifications,
  unregisterPushNotifications,
  isNotificationsEnabled,
  onNotification,
} from '../services/push-notifications';
import {sounds} from '../services/sounds';
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
  const theme = useTheme();
  const settings = useSettingsStore();
  const {health, refresh} = useModelStore();
  const navigation = useNavigation<any>();
  const [serverUrl, setServerUrl] = useState('');
  const serverUrlEditedRef = useRef(false);
  const [healthData, setHealthData] = useState<HealthStatus | null>(null);
  const [notificationsOn, setNotificationsOn] = useState(false);
  const [lastNotification, setLastNotification] = useState<string | null>(null);
  const [soundsOn, setSoundsOn] = useState(sounds.isEnabled());
  const hybrid = useHybridStore();

  useEffect(() => {
    getApiUrl().then(url => {
      if (!serverUrlEditedRef.current) {
        setServerUrl(url);
      }
    });
    api.get<HealthStatus>('/health').then(setHealthData).catch(() => {});
    isNotificationsEnabled().then(setNotificationsOn);
  }, []);

  useEffect(() => {
    return onNotification((title, body) => {
      setLastNotification(`${title}: ${body}`);
    });
  }, []);

  const handleToggleNotifications = async (val: boolean) => {
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
        <ScrollView contentContainerStyle={{padding: 16, gap: 12}}>
          <Text fontSize={22} fontWeight="700" letterSpacing={-0.3} color="$color" marginBottom={4}>
            Settings
          </Text>

          {/* Server */}
          <YStack backgroundColor="$background" borderRadius={12} borderWidth={0.5} borderColor="$borderColor" padding={16} gap={8}>
            <Text fontSize={15} fontWeight="600" color="$color">Server</Text>
            <XStack justifyContent="space-between" alignItems="center">
              <Text fontSize={14} color="$color11">Status</Text>
              <StatusBadge
                label={healthData?.status === 'healthy' ? 'Connected' : 'Offline'}
                variant={healthData?.status === 'healthy' ? 'success' : 'error'}
              />
            </XStack>
            <XStack justifyContent="space-between" alignItems="center">
              <Text fontSize={14} color="$color11">Model</Text>
              <Text fontSize={14} fontWeight="500" color="$color">{healthData?.model_name || 'None'}</Text>
            </XStack>
            <XStack gap={8} marginTop={8}>
              <Input
                flex={1}
                size="$3"
                value={serverUrl}
                onChangeText={val => { serverUrlEditedRef.current = true; setServerUrl(val); }}
                placeholder="http://localhost:8000"
                autoCapitalize="none"
                autoCorrect={false}
              />
              <Button size="$3" backgroundColor="$color9" color="white" fontWeight="600" onPress={handleSaveUrl}>Save</Button>
            </XStack>
          </YStack>

          {/* System Health nav */}
          <YStack backgroundColor="$background" borderRadius={12} borderWidth={0.5} borderColor="$borderColor" padding={16} onPress={() => navigation.navigate('Health')}>
            <XStack justifyContent="space-between" alignItems="center">
              <YStack>
                <Text fontSize={15} fontWeight="600" color="$color">System Health</Text>
                <Text fontSize={12} color="$color11">CPU, memory, disk, uptime</Text>
              </YStack>
              <Text fontSize={18} color="$color11" fontWeight="300">→</Text>
            </XStack>
          </YStack>

          {/* Inference */}
          <YStack backgroundColor="$background" borderRadius={12} borderWidth={0.5} borderColor="$borderColor" padding={16} gap={8}>
            <Text fontSize={15} fontWeight="600" color="$color">Inference</Text>

            {/* Engine chips */}
            <XStack gap={8}>
              {(['slonet', 'qwen', 'remote'] as const).map(engine => {
                const active = hybrid.activeEngine === engine;
                const label = engine === 'slonet' ? 'SloNet' : engine === 'qwen' ? 'Qwen' : 'Server';
                return (
                  <YStack
                    key={engine}
                    flex={1}
                    paddingVertical={8}
                    borderRadius={999}
                    backgroundColor={active ? '$color9' : '$backgroundHover'}
                    borderWidth={0.5}
                    borderColor={active ? '$color9' : '$borderColor'}
                    alignItems="center"
                    onPress={() => hybrid.setActiveEngine(engine)}>
                    <Text fontSize={11} fontWeight="600" color={active ? 'white' : '$color11'}>{label}</Text>
                  </YStack>
                );
              })}
            </XStack>

            {/* SloNet */}
            <XStack justifyContent="space-between" alignItems="center" paddingVertical={4}>
              <YStack>
                <Text fontSize={14} fontWeight="500" color="$color">SloNet</Text>
                <Text fontSize={12} color="$color11">
                  {hybrid.slonet.loaded ? `Loaded — ${hybrid.slonet.modelName}` : 'Not loaded'}
                </Text>
              </YStack>
              {hybrid.slonet.loaded ? (
                <Button size="$2" backgroundColor="$backgroundHover" color="$color11" fontWeight="600" borderRadius={999} onPress={hybrid.unloadSloNet}>Unload</Button>
              ) : hybrid.slonet.downloadProgress !== null && hybrid.slonet.downloadProgress < 1 ? (
                <ActivityIndicator size="small" color="$color9" />
              ) : (
                <Button size="$2" backgroundColor="$color9" color="white" fontWeight="600" borderRadius={999} onPress={() => hybrid.loadSloNet()}>Load</Button>
              )}
            </XStack>

            {/* Qwen */}
            <XStack justifyContent="space-between" alignItems="center" paddingVertical={4}>
              <YStack>
                <Text fontSize={14} fontWeight="500" color="$color">Qwen 0.5B GGUF</Text>
                <Text fontSize={12} color="$color11">
                  {hybrid.qwen.loaded
                    ? 'Loaded — 15-30 tok/s'
                    : hybrid.qwen.downloadProgress !== null && hybrid.qwen.downloadProgress < 1
                    ? `Downloading ${Math.round(hybrid.qwen.downloadProgress * 100)}%`
                    : 'Not downloaded'}
                </Text>
              </YStack>
              {hybrid.qwen.loaded ? (
                <Button size="$2" backgroundColor="$backgroundHover" color="$color11" fontWeight="600" borderRadius={999} onPress={() => hybrid.unloadQwen()}>Unload</Button>
              ) : hybrid.qwen.downloadProgress !== null && hybrid.qwen.downloadProgress < 1 ? (
                <YStack width={60} height={4} borderRadius={2} backgroundColor="$borderColor" overflow="hidden">
                  <YStack height="100%" backgroundColor="$color9" borderRadius={2} width={`${hybrid.qwen.downloadProgress * 100}%`} />
                </YStack>
              ) : (
                <Button size="$2" backgroundColor="$color9" color="white" fontWeight="600" borderRadius={999} onPress={() => hybrid.loadQwen()}>Download</Button>
              )}
            </XStack>

            {hybrid.lastError && (
              <Text fontSize={11} color="$color10">{hybrid.lastError}</Text>
            )}

            {/* Offline toggle */}
            <XStack gap={12} marginTop={8} paddingTop={12} borderTopWidth={1} borderTopColor="$borderColor" alignItems="center">
              <YStack flex={1}>
                <Text fontSize={15} fontWeight="600" color="$color">Run Completely Offline</Text>
                <Text fontSize={12} color="$color11">
                  {hybrid.offlineOnly
                    ? 'Server disabled — conversations use local engines only'
                    : 'Disables server fallback — load SloNet or Qwen first'}
                </Text>
              </YStack>
              <YStack
                paddingHorizontal={12} paddingVertical={8} borderRadius={999}
                backgroundColor={hybrid.offlineOnly ? '$background' : '$backgroundHover'}
                borderWidth={0.5}
                borderColor={hybrid.offlineOnly ? '$color9' : '$borderColor'}
                minWidth={56} alignItems="center"
                onPress={() => hybrid.setOfflineOnly(!hybrid.offlineOnly)}>
                <Text fontSize={11} fontWeight="600" color={hybrid.offlineOnly ? '$color9' : '$color11'}>
                  {hybrid.offlineOnly ? 'ON' : 'OFF'}
                </Text>
              </YStack>
            </XStack>
          </YStack>

          {/* Appearance */}
          <YStack backgroundColor="$background" borderRadius={12} borderWidth={0.5} borderColor="$borderColor" padding={16} gap={8}>
            <Text fontSize={15} fontWeight="600" color="$color">Appearance</Text>
            <XStack gap={8}>
              {themes.map(t => {
                const active = settings.theme === t;
                return (
                  <YStack
                    key={t}
                    flex={1}
                    paddingVertical={10}
                    borderRadius={999}
                    backgroundColor={active ? '$color9' : '$backgroundHover'}
                    borderWidth={0.5}
                    borderColor={active ? '$color9' : '$borderColor'}
                    alignItems="center"
                    onPress={() => settings.setTheme(t)}>
                    <Text fontSize={11} fontWeight="600" color={active ? 'white' : '$color11'}>
                      {t.charAt(0).toUpperCase() + t.slice(1)}
                    </Text>
                  </YStack>
                );
              })}
            </XStack>
            <XStack gap={8} marginTop={4}>
              {([
                {key: 'violet', color: '#7C52C4', label: 'Violet'},
                {key: 'rose', color: '#E11D48', label: 'Rose'},
                {key: 'amber', color: '#D97706', label: 'Amber'},
                {key: 'emerald', color: '#059669', label: 'Emerald'},
                {key: 'sky', color: '#0284C7', label: 'Sky'},
              ] as const).map(opt => {
                const active = settings.accentColor === opt.key;
                return (
                  <YStack
                    key={opt.key}
                    width={36}
                    height={36}
                    borderRadius={999}
                    backgroundColor={opt.color}
                    borderWidth={2}
                    borderColor={active ? opt.color : 'transparent'}
                    alignItems="center"
                    justifyContent="center"
                    onPress={() => settings.update({accentColor: opt.key})}>
                    {active && (
                      <YStack width={12} height={12} borderRadius={6} backgroundColor="white" />
                    )}
                  </YStack>
                );
              })}
            </XStack>
          </YStack>

          {/* Font */}
          <YStack backgroundColor="$background" borderRadius={12} borderWidth={0.5} borderColor="$borderColor" padding={16} gap={8}>
            <Text fontSize={15} fontWeight="600" color="$color">Font</Text>
            <Text fontSize={12} color="$color11">Typeface</Text>
            <XStack gap={8}>
              {FONT_FAMILY_OPTIONS.map(opt => {
                const active = settings.fontFamily === opt.value;
                return (
                  <YStack
                    key={opt.value}
                    flex={1}
                    paddingVertical={10}
                    borderRadius={999}
                    backgroundColor={active ? '$color9' : '$backgroundHover'}
                    borderWidth={0.5}
                    borderColor={active ? '$color9' : '$borderColor'}
                    alignItems="center"
                    onPress={() => settings.setFontFamily(opt.value)}>
                    <Text fontSize={11} fontWeight="600" color={active ? 'white' : '$color11'}>{opt.label}</Text>
                  </YStack>
                );
              })}
            </XStack>
            <Text fontSize={12} color="$color11" marginTop={4}>Size</Text>
            <XStack gap={4} flexWrap="wrap">
              {FONT_SCALE_PRESETS.map(p => {
                const active = settings.fontSizeScale === p.value;
                return (
                  <YStack
                    key={p.value}
                    paddingHorizontal={12} paddingVertical={6}
                    borderRadius={999}
                    backgroundColor={active ? '$color9' : '$backgroundHover'}
                    borderWidth={0.5}
                    borderColor={active ? '$color9' : '$borderColor'}
                    onPress={() => settings.setFontSizeScale(p.value)}>
                    <Text fontSize={11} fontWeight="500" color={active ? 'white' : '$color11'}>{p.label}</Text>
                  </YStack>
                );
              })}
            </XStack>
          </YStack>

          {/* Push Notifications */}
          <YStack backgroundColor="$background" borderRadius={12} borderWidth={0.5} borderColor="$borderColor" padding={16}>
            <XStack justifyContent="space-between" alignItems="center">
              <YStack flex={1}>
                <Text fontSize={15} fontWeight="600" color="$color">Push Notifications</Text>
                <Text fontSize={12} color="$color11">Training updates and chat messages</Text>
              </YStack>
              <Switch
                value={notificationsOn}
                onValueChange={handleToggleNotifications}
                trackColor={{false: theme.borderColor?.val || '#E4E0F2', true: (theme.color9?.val || '#7C52C4') + '60'}}
                thumbColor={notificationsOn ? (theme.color9?.val || '#7C52C4') : (theme.color10?.val || '#827A96')}
              />
            </XStack>
            {lastNotification && (
              <YStack backgroundColor="$background" marginTop={8} borderRadius={4} padding={8}>
                <Text fontSize={11} color="$color11">{lastNotification}</Text>
              </YStack>
            )}
          </YStack>

          {/* Sound Effects */}
          <YStack backgroundColor="$background" borderRadius={12} borderWidth={0.5} borderColor="$borderColor" padding={16}>
            <XStack justifyContent="space-between" alignItems="center">
              <YStack flex={1}>
                <Text fontSize={15} fontWeight="600" color="$color">Sound Effects</Text>
                <Text fontSize={12} color="$color11">Audio feedback on send/receive</Text>
              </YStack>
              <Switch
                value={soundsOn}
                onValueChange={(val) => { setSoundsOn(val); sounds.setEnabled(val); }}
                trackColor={{false: theme.borderColor?.val || '#E4E0F2', true: (theme.color9?.val || '#7C52C4') + '60'}}
                thumbColor={soundsOn ? (theme.color9?.val || '#7C52C4') : (theme.color10?.val || '#827A96')}
              />
            </XStack>
          </YStack>

          {/* Chat Defaults */}
          <YStack backgroundColor="$background" borderRadius={12} borderWidth={0.5} borderColor="$borderColor" padding={16} gap={8}>
            <Text fontSize={15} fontWeight="600" color="$color">Chat Defaults</Text>
            {[
              {label: 'Temperature', value: settings.temperature.toFixed(1), key: 'temperature', options: [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]},
              {label: 'Max Tokens', value: String(settings.maxTokens), key: 'maxTokens', options: [128, 256, 512, 1024], exact: true},
              {label: 'Top-P', value: settings.topP.toFixed(1), key: 'topP', options: [0.7, 0.8, 0.9, 1.0]},
              {label: 'Top-K', value: String(settings.topK), key: 'topK', options: [20, 50, 100, 200], exact: true},
              {label: 'Repetition Penalty', value: settings.repetitionPenalty.toFixed(1), key: 'repetitionPenalty', options: [1.0, 1.1, 1.2, 1.5, 2.0]},
            ].map(({label, value, key, options, exact}) => (
              <YStack key={key} gap={4}>
                <Text fontSize={13} color="$color11">{label}: {value}</Text>
                <XStack gap={4} flexWrap="wrap">
                  {options.map(v => {
                    const match = exact
                      ? (settings as any)[key] === v
                      : Math.abs((settings as any)[key] - v) < 0.05;
                    return (
                      <YStack
                        key={String(v)}
                        paddingHorizontal={12} paddingVertical={6}
                        borderRadius={999}
                        backgroundColor={match ? '$color9' : '$backgroundHover'}
                        borderWidth={0.5}
                        borderColor={match ? '$color9' : '$borderColor'}
                        onPress={() => settings.update({[key]: v})}>
                        <Text fontSize={11} fontWeight="500" color={match ? 'white' : '$color11'}>
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
          <YStack backgroundColor="$background" borderRadius={12} borderWidth={0.5} borderColor="$borderColor" padding={16} gap={8}>
            <Text fontSize={15} fontWeight="600" color="$color">Chat Background</Text>
            <Text fontSize={12} color="$color11">Custom tint behind messages</Text>
            <XStack gap={6} flexWrap="wrap" marginTop={4}>
              {[
                {label: 'Default', value: ''},
                {label: 'Warm', value: 'warm'},
                {label: 'Cool', value: 'cool'},
                {label: 'Lavender', value: 'lavender'},
                {label: 'Peach', value: 'peach'},
              ].map(({label, value}) => {
                const active = settings.chatBackground === value;
                return (
                  <YStack
                    key={value}
                    paddingHorizontal={14} paddingVertical={7}
                    borderRadius={999}
                    backgroundColor={active ? '$color9' : '$backgroundHover'}
                    borderWidth={0.5}
                    borderColor={active ? '$color9' : '$borderColor'}
                    onPress={() => settings.update({chatBackground: value})}>
                    <Text fontSize={12} fontWeight="500" color={active ? 'white' : '$color11'}>
                      {label}
                    </Text>
                  </YStack>
                );
              })}
            </XStack>
          </YStack>

          {/* Chat Background */}
          <YStack backgroundColor="$background" borderRadius={12} borderWidth={0.5} borderColor="$borderColor" padding={16} gap={8}>
            <Text fontSize={15} fontWeight="600" color="$color">Chat Background</Text>
            <Text fontSize={12} color="$color11">Customize the chat area tint</Text>
            <XStack gap={8} flexWrap="wrap" marginTop={4}>
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
                    borderRadius={999}
                    backgroundColor={active ? '$color9' : '$backgroundHover'}
                    borderWidth={0.5}
                    borderColor={active ? '$color9' : '$borderColor'}
                    onPress={() => settings.update({chatBackground: opt.value})}>
                    <Text fontSize={12} fontWeight="500" color={active ? 'white' : '$color11'}>
                      {opt.label}
                    </Text>
                  </YStack>
                );
              })}
            </XStack>
          </YStack>

          {/* Memory Context */}
          <YStack backgroundColor="$background" borderRadius={12} borderWidth={0.5} borderColor="$borderColor" padding={16} gap={8}>
            <Text fontSize={15} fontWeight="600" color="$color">Memory Context</Text>
            <RNTextInput
              style={{
                fontSize: 14, color: theme.color?.val || '#1A1625', backgroundColor: theme.background?.val || '#FFFFFF',
                borderRadius: 8, padding: 12, minHeight: 80,
                borderWidth: 1, borderColor: theme.borderColor?.val || '#E4E0F2', textAlignVertical: 'top',
              }}
              value={settings.memoryContext}
              onChangeText={v => settings.update({memoryContext: v})}
              placeholder="Custom context the AI always remembers..."
              placeholderTextColor={theme.color10?.val || '#827A96'}
              multiline
            />
          </YStack>

          {/* Nav cards */}
          {[
            {label: 'Bookmarks', desc: 'Saved messages for quick access', target: 'Bookmarks'},
            {label: 'About SloughGPT', desc: 'Version, features, architecture', target: 'About'},
            {label: 'Training', desc: 'Fine-tune models by chatting and interacting', target: 'Training'},
            {label: 'What AI Knows About Me', desc: "View and manage the AI's knowledge about you", target: 'Knowledge'},
            {label: 'Help', desc: 'FAQ, keyboard shortcuts, troubleshooting', target: 'Help'},
            {label: 'Search Messages', desc: 'Find messages across conversations', target: 'Search'},
          ].map(({label, desc, target}) => (
            <YStack
              key={target}
              backgroundColor="$background" borderRadius={12} borderWidth={0.5} borderColor="$borderColor" padding={16}
              onPress={() => navigation.navigate(target)}>
              <XStack justifyContent="space-between" alignItems="center">
                <YStack>
                  <Text fontSize={15} fontWeight="600" color="$color">{label}</Text>
                  <Text fontSize={12} color="$color11">{desc}</Text>
                </YStack>
                <Text fontSize={18} color="$color11" fontWeight="300">→</Text>
              </XStack>
            </YStack>
          ))}

          {/* Danger Zone */}
          <YStack backgroundColor="$background" borderRadius={12} borderWidth={0.5} borderColor="$borderColor" padding={16} gap={8}>
            <Text fontSize={15} fontWeight="600" color="$color">Danger Zone</Text>
            <YStack
              paddingVertical={10} paddingHorizontal={12} borderRadius={999}
              backgroundColor="#FEF2F2" alignItems="center"
              borderWidth={0.5} borderColor="#FEE2E2"
              onPress={() =>
                Alert.alert('Reset Settings', 'Reset all settings to defaults?', [
                  {text: 'Cancel', style: 'cancel'},
                  {text: 'Reset', style: 'destructive', onPress: settings.reset},
                ])
              }>
              <Text fontSize={11} fontWeight="600" color="$color10">Reset all settings</Text>
            </YStack>
          </YStack>

          <YStack alignItems="center" paddingVertical={24}>
            <Text fontSize={11} color="$color11" opacity={0.4}>SloughGPT v1.0.0</Text>
          </YStack>
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}
