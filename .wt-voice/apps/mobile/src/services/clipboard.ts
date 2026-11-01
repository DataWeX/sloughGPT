import {Platform, Clipboard, Alert, Linking} from 'react-native';
import {triggerHaptic} from './haptics';

export async function copyToClipboard(text: string): Promise<boolean> {
  try {
    Clipboard.setString(text);
    triggerHaptic('success');
    return true;
  } catch {
    return false;
  }
}

export async function copyWithFeedback(
  text: string,
  label?: string,
): Promise<void> {
  const ok = await copyToClipboard(text);
  if (ok) {
    triggerHaptic('success');
  }
}

export function permissionDeniedAlert(permission: string): void {
  Alert.alert(
    `${permission} permission required`,
    `Please enable ${permission.toLowerCase()} access in your device settings to use this feature.`,
    [
      {text: 'Cancel', style: 'cancel'},
      {
        text: 'Open Settings',
        onPress: () => Linking.openSettings().catch(() => {}),
      },
    ],
  );
}
