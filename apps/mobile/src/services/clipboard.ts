import {Platform, Clipboard} from 'react-native';

export async function copyToClipboard(text: string): Promise<boolean> {
  try {
    if (Platform.OS === 'ios') {
      Clipboard.setString(text);
      return true;
    }
    Clipboard.setString(text);
    return true;
  } catch {
    return false;
  }
}
