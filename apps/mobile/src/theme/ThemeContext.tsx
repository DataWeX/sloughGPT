import React, {createContext, useContext, useMemo} from 'react';
import {useColorScheme} from 'react-native';
import {useSettingsStore} from '../stores/settings-store';
import {
  colors as lightColors,
  darkColors,
  spacing,
  radii,
  typography,
  layout,
} from '../theme';

type ThemeColors = typeof lightColors;

interface ThemeContextValue {
  colors: ThemeColors;
  spacing: typeof spacing;
  radii: typeof radii;
  typography: typeof typography;
  layout: typeof layout;
  isDark: boolean;
}

const ThemeContext = createContext<ThemeContextValue>({
  colors: lightColors,
  spacing,
  radii,
  typography,
  layout,
  isDark: false,
});

export function ThemeProvider({children}: {children: React.ReactNode}) {
  const systemScheme = useColorScheme();
  const themeMode = useSettingsStore(s => s.theme);

  const isDark = useMemo(() => {
    if (themeMode === 'system') return systemScheme === 'dark';
    return themeMode === 'dark';
  }, [themeMode, systemScheme]);

  const value = useMemo(
    () => ({
      colors: isDark ? darkColors : lightColors,
      spacing,
      radii,
      typography,
      layout,
      isDark,
    }),
    [isDark],
  );

  return (
    <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
  );
}

export function useTheme() {
  return useContext(ThemeContext);
}
