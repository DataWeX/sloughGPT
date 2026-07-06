import React, {useEffect, useState} from 'react';
import {
  View,
  Text,
  ScrollView,
  TouchableOpacity,
  TextInput,
  StyleSheet,
  Alert,
  Switch,
  ActivityIndicator,
} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {useNavigation} from '@react-navigation/native';
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
import {colors, spacing, radii, typography} from '../theme';
import type {HealthStatus} from '../types';
import type {ThemeMode} from '../types';

export function SettingsScreen() {
  const settings = useSettingsStore();
  const {health, refresh} = useModelStore();
  const navigation = useNavigation<any>();
  const [serverUrl, setServerUrl] = useState('');
  const [healthData, setHealthData] = useState<HealthStatus | null>(null);
  const [notificationsOn, setNotificationsOn] = useState(false);
  const [lastNotification, setLastNotification] = useState<string | null>(null);
  const [soundsOn, setSoundsOn] = useState(sounds.isEnabled());
  const hybrid = useHybridStore();

  useEffect(() => {
    getApiUrl().then(setServerUrl);
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
    <SafeAreaView style={styles.safe} edges={['top']}>
      <ScrollView contentContainerStyle={styles.content}>
        <Text style={styles.title}>Settings</Text>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Server</Text>
          <View style={styles.row}>
            <Text style={styles.label}>Status</Text>
            <StatusBadge
              label={healthData?.status === 'healthy' ? 'Connected' : 'Offline'}
              variant={
                healthData?.status === 'healthy' ? 'success' : 'error'
              }
            />
          </View>
          <View style={styles.row}>
            <Text style={styles.label}>Model</Text>
            <Text style={styles.value}>
              {healthData?.model_name || 'None'}
            </Text>
          </View>
          <View style={styles.urlRow}>
            <TextInput
              style={styles.urlInput}
              value={serverUrl}
              onChangeText={setServerUrl}
              placeholder="http://localhost:8000"
              placeholderTextColor={colors.textMuted}
              autoCapitalize="none"
              autoCorrect={false}
            />
            <TouchableOpacity style={styles.urlSaveBtn} onPress={handleSaveUrl}>
              <Text style={styles.urlSaveText}>Save</Text>
            </TouchableOpacity>
          </View>
        </View>

        <TouchableOpacity
          style={styles.card}
          onPress={() => navigation.navigate('Health')}>
          <View style={styles.navRow}>
            <View>
              <Text style={styles.cardTitle}>System Health</Text>
              <Text style={styles.navDesc}>CPU, memory, disk, uptime</Text>
            </View>
            <Text style={styles.navArrow}>→</Text>
          </View>
        </TouchableOpacity>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Inference</Text>

          {/* Engine selector */}
          <View style={styles.inferenceChips}>
            {(['slonet', 'qwen', 'remote'] as const).map(engine => {
              const active = hybrid.activeEngine === engine;
              const label =
                engine === 'slonet' ? 'SloNet' : engine === 'qwen' ? 'Qwen' : 'Server';
              return (
                <TouchableOpacity
                  key={engine}
                  style={[styles.inferenceChip, active && styles.inferenceChipActive]}
                  onPress={() => hybrid.setActiveEngine(engine)}>
                  <Text
                    style={[
                      styles.inferenceChipText,
                      active && styles.inferenceChipTextActive,
                    ]}>
                    {label}
                  </Text>
                </TouchableOpacity>
              );
            })}
          </View>

          {/* SloNet status */}
          <View style={styles.inferenceRow}>
            <View style={{flex: 1}}>
              <Text style={styles.inferenceName}>SloNet</Text>
              <Text style={styles.inferenceMeta}>
                {hybrid.slonet.loaded
                  ? `Loaded — ${hybrid.slonet.modelName}`
                  : 'Not loaded'}
              </Text>
            </View>
            {hybrid.slonet.loaded ? (
              <TouchableOpacity style={styles.inferenceUnloadBtn} onPress={hybrid.unloadSloNet}>
                <Text style={styles.inferenceUnloadText}>Unload</Text>
              </TouchableOpacity>
            ) : hybrid.slonet.downloadProgress !== null && hybrid.slonet.downloadProgress < 1 ? (
              <ActivityIndicator size="small" color={colors.primary} />
            ) : (
              <TouchableOpacity style={styles.inferenceLoadBtn} onPress={() => hybrid.loadSloNet()}>
                <Text style={styles.inferenceLoadText}>Load</Text>
              </TouchableOpacity>
            )}
          </View>

          {/* Qwen status */}
          <View style={styles.inferenceRow}>
            <View style={{flex: 1}}>
              <Text style={styles.inferenceName}>Qwen 0.5B GGUF</Text>
              <Text style={styles.inferenceMeta}>
                {hybrid.qwen.loaded
                  ? 'Loaded — 15-30 tok/s'
                  : hybrid.qwen.downloadProgress !== null && hybrid.qwen.downloadProgress < 1
                  ? `Downloading ${Math.round(hybrid.qwen.downloadProgress * 100)}%`
                  : 'Not downloaded'}
              </Text>
            </View>
            {hybrid.qwen.loaded ? (
              <TouchableOpacity style={styles.inferenceUnloadBtn} onPress={() => hybrid.unloadQwen()}>
                <Text style={styles.inferenceUnloadText}>Unload</Text>
              </TouchableOpacity>
            ) : hybrid.qwen.downloadProgress !== null && hybrid.qwen.downloadProgress < 1 ? (
              <View style={styles.inferenceProgressWrap}>
                <View style={[styles.inferenceProgressFill, {width: `${hybrid.qwen.downloadProgress * 100}%`}]} />
              </View>
            ) : (
              <TouchableOpacity style={styles.inferenceLoadBtn} onPress={() => hybrid.loadQwen()}>
                <Text style={styles.inferenceLoadText}>Download</Text>
              </TouchableOpacity>
            )}
          </View>

          {hybrid.lastError && (
            <Text style={styles.inferenceError}>{hybrid.lastError}</Text>
          )}

          <View style={styles.offlineRow}>
            <View style={{flex: 1}}>
              <Text style={styles.cardTitle}>Run Completely Offline</Text>
              <Text style={styles.navDesc}>
                {hybrid.offlineOnly
                  ? 'Server disabled — conversations use local engines only'
                  : 'Disables server fallback — load SloNet or Qwen first'}
              </Text>
            </View>
            <TouchableOpacity
              style={[
                styles.offlineToggle,
                hybrid.offlineOnly && styles.offlineToggleActive,
              ]}
              onPress={() => hybrid.setOfflineOnly(!hybrid.offlineOnly)}>
              <Text
                style={[
                  styles.offlineToggleText,
                  hybrid.offlineOnly && styles.offlineToggleTextActive,
                ]}>
                {hybrid.offlineOnly ? 'ON' : 'OFF'}
              </Text>
            </TouchableOpacity>
          </View>
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Appearance</Text>
          <View style={styles.themeRow}>
            {themes.map(t => (
              <TouchableOpacity
                key={t}
                style={[
                  styles.themeBtn,
                  settings.theme === t && styles.themeBtnActive,
                ]}
                onPress={() => settings.setTheme(t)}>
                <Text
                  style={[
                    styles.themeBtnText,
                    settings.theme === t && styles.themeBtnTextActive,
                  ]}>
                  {t.charAt(0).toUpperCase() + t.slice(1)}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>

        <View style={styles.card}>
          <View style={styles.row}>
            <View style={{flex: 1}}>
              <Text style={styles.cardTitle}>Push Notifications</Text>
              <Text style={styles.navDesc}>Training updates and chat messages</Text>
            </View>
            <Switch
              value={notificationsOn}
              onValueChange={handleToggleNotifications}
              trackColor={{false: colors.border, true: colors.primary + '60'}}
              thumbColor={notificationsOn ? colors.primary : colors.textMuted}
            />
          </View>
          {lastNotification && (
            <View style={[styles.card, {backgroundColor: colors.surface, marginTop: 8, borderRadius: radii.sm}]}>
              <Text style={[styles.navDesc, {fontSize: 11}]}>{lastNotification}</Text>
            </View>
          )}
        </View>

        <View style={styles.card}>
          <View style={styles.row}>
            <View style={{flex: 1}}>
              <Text style={styles.cardTitle}>Sound Effects</Text>
              <Text style={styles.navDesc}>Audio feedback on send/receive</Text>
            </View>
            <Switch
              value={soundsOn}
              onValueChange={(val) => {
                setSoundsOn(val);
                sounds.setEnabled(val);
              }}
              trackColor={{false: colors.border, true: colors.primary + '60'}}
              thumbColor={soundsOn ? colors.primary : colors.textMuted}
            />
          </View>
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Chat Defaults</Text>
          <View style={styles.sliderRow}>
            <Text style={styles.label}>
              Temperature: {settings.temperature.toFixed(1)}
            </Text>
            <View style={styles.sliderBtns}>
              {[0.2, 0.4, 0.6, 0.8, 1.0, 1.2].map(v => (
                <TouchableOpacity
                  key={v}
                  style={[
                    styles.tempBtn,
                    Math.abs(settings.temperature - v) < 0.05 &&
                      styles.tempBtnActive,
                  ]}
                  onPress={() => settings.update({temperature: v})}>
                  <Text
                    style={[
                      styles.tempBtnText,
                      Math.abs(settings.temperature - v) < 0.05 &&
                        styles.tempBtnTextActive,
                    ]}>
                    {v.toFixed(1)}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
          </View>
          <View style={styles.sliderRow}>
            <Text style={styles.label}>
              Max Tokens: {settings.maxTokens}
            </Text>
            <View style={styles.sliderBtns}>
              {[128, 256, 512, 1024].map(v => (
                <TouchableOpacity
                  key={v}
                  style={[
                    styles.tempBtn,
                    settings.maxTokens === v && styles.tempBtnActive,
                  ]}
                  onPress={() => settings.update({maxTokens: v})}>
                  <Text
                    style={[
                      styles.tempBtnText,
                      settings.maxTokens === v && styles.tempBtnTextActive,
                    ]}>
                    {v}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
          </View>
          <View style={styles.sliderRow}>
            <Text style={styles.label}>
              Top-P: {settings.topP.toFixed(1)}
            </Text>
            <View style={styles.sliderBtns}>
              {[0.7, 0.8, 0.9, 1.0].map(v => (
                <TouchableOpacity
                  key={v}
                  style={[
                    styles.tempBtn,
                    Math.abs(settings.topP - v) < 0.05 && styles.tempBtnActive,
                  ]}
                  onPress={() => settings.update({topP: v})}>
                  <Text
                    style={[
                      styles.tempBtnText,
                      Math.abs(settings.topP - v) < 0.05 &&
                        styles.tempBtnTextActive,
                    ]}>
                    {v.toFixed(1)}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
          </View>
          <View style={styles.sliderRow}>
            <Text style={styles.label}>
              Top-K: {settings.topK}
            </Text>
            <View style={styles.sliderBtns}>
              {[20, 50, 100, 200].map(v => (
                <TouchableOpacity
                  key={v}
                  style={[
                    styles.tempBtn,
                    settings.topK === v && styles.tempBtnActive,
                  ]}
                  onPress={() => settings.update({topK: v})}>
                  <Text
                    style={[
                      styles.tempBtnText,
                      settings.topK === v && styles.tempBtnTextActive,
                    ]}>
                    {v}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
          </View>
          <View style={styles.sliderRow}>
            <Text style={styles.label}>
              Repetition Penalty: {settings.repetitionPenalty.toFixed(1)}
            </Text>
            <View style={styles.sliderBtns}>
              {[1.0, 1.1, 1.2, 1.5, 2.0].map(v => (
                <TouchableOpacity
                  key={v}
                  style={[
                    styles.tempBtn,
                    Math.abs(settings.repetitionPenalty - v) < 0.05 &&
                      styles.tempBtnActive,
                  ]}
                  onPress={() => settings.update({repetitionPenalty: v})}>
                  <Text
                    style={[
                      styles.tempBtnText,
                      Math.abs(settings.repetitionPenalty - v) < 0.05 &&
                        styles.tempBtnTextActive,
                    ]}>
                    {v.toFixed(1)}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
          </View>
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Memory Context</Text>
          <TextInput
            style={styles.textArea}
            value={settings.memoryContext}
            onChangeText={v => settings.update({memoryContext: v})}
            placeholder="Custom context the AI always remembers..."
            placeholderTextColor={colors.textMuted}
            multiline
            textAlignVertical="top"
          />
        </View>

        <TouchableOpacity
          style={styles.card}
          onPress={() => navigation.navigate('Bookmarks')}>
          <View style={styles.navRow}>
            <View>
              <Text style={styles.cardTitle}>Bookmarks</Text>
              <Text style={styles.navDesc}>Saved messages for quick access</Text>
            </View>
            <Text style={styles.navArrow}>→</Text>
          </View>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.card}
          onPress={() => navigation.navigate('About')}>
          <View style={styles.navRow}>
            <View>
              <Text style={styles.cardTitle}>About SloughGPT</Text>
              <Text style={styles.navDesc}>Version, features, architecture</Text>
            </View>
            <Text style={styles.navArrow}>→</Text>
          </View>
        </TouchableOpacity>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Danger Zone</Text>
          <TouchableOpacity
            style={styles.dangerBtn}
            onPress={() =>
              Alert.alert('Reset Settings', 'Reset all settings to defaults?', [
                {text: 'Cancel', style: 'cancel'},
                {text: 'Reset', style: 'destructive', onPress: settings.reset},
              ])
            }>
            <Text style={styles.dangerText}>Reset all settings</Text>
          </TouchableOpacity>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: colors.background,
  },
  content: {
    padding: spacing.lg,
    gap: spacing.md,
  },
  title: {
    ...typography.h1,
    color: colors.text,
  },
  card: {
    backgroundColor: colors.surface,
    borderRadius: radii.lg,
    padding: spacing.lg,
  },
  cardTitle: {
    ...typography.h3,
    color: colors.text,
    marginBottom: spacing.md,
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: spacing.sm,
  },
  label: {
    ...typography.body,
    color: colors.textSecondary,
  },
  value: {
    ...typography.body,
    color: colors.text,
    fontWeight: '500',
  },
  urlRow: {
    flexDirection: 'row',
    gap: spacing.sm,
    marginTop: spacing.sm,
  },
  urlInput: {
    flex: 1,
    ...typography.caption,
    color: colors.text,
    backgroundColor: colors.background,
    borderRadius: radii.md,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
  },
  urlSaveBtn: {
    backgroundColor: colors.primary,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    borderRadius: radii.md,
    justifyContent: 'center',
  },
  urlSaveText: {
    ...typography.caption,
    color: colors.white,
    fontWeight: '600',
  },
  inferenceChips: {
    flexDirection: 'row',
    gap: spacing.xs,
    marginBottom: spacing.md,
  },
  inferenceChip: {
    flex: 1,
    paddingVertical: spacing.sm,
    borderRadius: radii.md,
    backgroundColor: colors.background,
    borderWidth: 1,
    borderColor: colors.border,
    alignItems: 'center',
  },
  inferenceChipActive: {
    backgroundColor: colors.primary,
    borderColor: colors.primary,
  },
  inferenceChipText: {
    ...typography.small,
    color: colors.textSecondary,
    fontWeight: '500',
  },
  inferenceChipTextActive: {
    color: colors.white,
  },
  inferenceRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: spacing.sm,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  inferenceName: {
    ...typography.body,
    color: colors.text,
    fontWeight: '500',
  },
  inferenceMeta: {
    ...typography.small,
    color: colors.textMuted,
    marginTop: 2,
  },
  inferenceLoadBtn: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.xs + 2,
    borderRadius: radii.md,
    backgroundColor: colors.primary,
  },
  inferenceLoadText: {
    ...typography.caption,
    color: colors.white,
    fontWeight: '600',
  },
  inferenceUnloadBtn: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.xs + 2,
    borderRadius: radii.md,
    backgroundColor: colors.error + '15',
  },
  inferenceUnloadText: {
    ...typography.caption,
    color: colors.error,
    fontWeight: '600',
  },
  inferenceProgressWrap: {
    width: 60,
    height: 6,
    borderRadius: 3,
    backgroundColor: colors.border,
    overflow: 'hidden',
  },
  inferenceProgressFill: {
    height: '100%',
    backgroundColor: colors.primary,
    borderRadius: 3,
  },
  inferenceError: {
    ...typography.small,
    color: colors.error,
    marginTop: spacing.sm,
  },
  offlineRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
    marginTop: spacing.md,
    paddingTop: spacing.md,
    borderTopWidth: 1,
    borderTopColor: colors.border,
  },
  offlineToggle: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    borderRadius: radii.md,
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.border,
    minWidth: 56,
    alignItems: 'center',
  },
  offlineToggleActive: {
    backgroundColor: colors.success + '20',
    borderColor: colors.success,
  },
  offlineToggleText: {
    ...typography.caption,
    color: colors.textSecondary,
    fontWeight: '600',
  },
  offlineToggleTextActive: {
    color: colors.success,
  },
  themeRow: {
    flexDirection: 'row',
    gap: spacing.sm,
  },
  themeBtn: {
    flex: 1,
    paddingVertical: spacing.sm,
    borderRadius: radii.md,
    backgroundColor: colors.background,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: colors.border,
  },
  themeBtnActive: {
    backgroundColor: colors.primary,
    borderColor: colors.primary,
  },
  themeBtnText: {
    ...typography.caption,
    color: colors.textSecondary,
    fontWeight: '500',
  },
  themeBtnTextActive: {
    color: colors.white,
  },
  sliderRow: {
    marginBottom: spacing.md,
  },
  sliderBtns: {
    flexDirection: 'row',
    gap: spacing.xs,
    marginTop: spacing.sm,
    flexWrap: 'wrap',
  },
  tempBtn: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.xs,
    borderRadius: radii.full,
    backgroundColor: colors.background,
    borderWidth: 1,
    borderColor: colors.border,
  },
  tempBtnActive: {
    backgroundColor: colors.primary,
    borderColor: colors.primary,
  },
  tempBtnText: {
    ...typography.small,
    color: colors.textSecondary,
  },
  tempBtnTextActive: {
    color: colors.white,
  },
  textArea: {
    ...typography.body,
    color: colors.text,
    backgroundColor: colors.background,
    borderRadius: radii.md,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.md,
    minHeight: 80,
  },
  dangerBtn: {
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.md,
    backgroundColor: colors.error + '15',
    borderRadius: radii.md,
    alignItems: 'center',
  },
  dangerText: {
    ...typography.caption,
    color: colors.error,
    fontWeight: '600',
  },
  navRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  navDesc: {
    ...typography.caption,
    color: colors.textMuted,
    marginTop: 2,
  },
  navArrow: {
    fontSize: 20,
    color: colors.textMuted,
  },
});
