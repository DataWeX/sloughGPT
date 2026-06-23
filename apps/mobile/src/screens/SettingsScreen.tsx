import React, {useEffect, useState} from 'react';
import {
  View,
  Text,
  ScrollView,
  TouchableOpacity,
  TextInput,
  StyleSheet,
  Alert,
} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {useNavigation} from '@react-navigation/native';
import {useSettingsStore} from '../stores/settings-store';
import {useModelStore} from '../stores/model-store';
import {StatusBadge} from '../components/StatusBadge';
import {api, getApiUrl, setApiUrl} from '../services/api-client';
import {colors, spacing, radii, typography} from '../theme';
import type {HealthStatus} from '../types';
import type {ThemeMode} from '../types';

export function SettingsScreen() {
  const settings = useSettingsStore();
  const {health, refresh} = useModelStore();
  const navigation = useNavigation<any>();
  const [serverUrl, setServerUrl] = useState('');
  const [healthData, setHealthData] = useState<HealthStatus | null>(null);

  useEffect(() => {
    getApiUrl().then(setServerUrl);
    api.get<HealthStatus>('/health').then(setHealthData).catch(() => {});
  }, []);

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
