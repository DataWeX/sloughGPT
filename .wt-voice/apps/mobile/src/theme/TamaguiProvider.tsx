import React from 'react';
import {TamaguiProvider as TProvider} from 'tamagui';
import config from '../../tamagui.config';
import {useSettingsStore} from '../stores/settings-store';
import {useColorScheme} from 'react-native';

export function TamaguiProvider({children}: {children: React.ReactNode}) {
  const systemScheme = useColorScheme();
  const themeMode = useSettingsStore(s => s.theme);

  const tamaguiTheme =
    themeMode === 'dark' || (themeMode === 'system' && systemScheme === 'dark')
      ? 'dark'
      : 'light';

  return (
    <TProvider config={config} defaultTheme={tamaguiTheme}>
      {children}
    </TProvider>
  );
}
