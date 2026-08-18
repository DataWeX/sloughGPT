import {useTheme, useThemeName} from 'tamagui';

export function useColors() {
  const theme = useTheme();
  const themeName = useThemeName();
  const isDark = themeName === 'dark';

  return {
    primary: (theme.color9?.val as string) || '#7C52C4',
    primaryLight: isDark ? '#C0AAF4' : '#7C52C4',
    text: (theme.color?.val as string) || (isDark ? '#F0ECF5' : '#1A1625'),
    textMuted: (theme.placeholderColor?.val as string) || '#827A96',
    textSecondary: (theme.color10?.val as string) || '#9B95A8',
    textOnPrimary: '#FFFFFF',
    background: (theme.background?.val as string) || (isDark ? '#110F18' : '#F8F6FC'),
    backgroundHover: (theme.backgroundHover?.val as string) || (isDark ? '#1C1926' : '#F0EDFA'),
    border: (theme.borderColor?.val as string) || (isDark ? '#342E48' : '#E4E0F2'),
    error: '#EF4444',
    errorDark: '#D44C56',
    errorLight: '#FDE8E8',
    success: '#34B07D',
    successDark: '#2E9B7C',
    successLight: '#E8F5EE',
    warning: '#F59E0B',
    warningDark: '#E8A83C',
    white: '#FFFFFF',
    overlay: (opacity: number) => `rgba(0, 0, 0, ${opacity})`,
    primaryAlpha: (opacity: number) => `rgba(124, 82, 196, ${opacity})`,
    errorAlpha: (opacity: number) => `rgba(239, 68, 68, ${opacity})`,
    errorDarkAlpha: (opacity: number) => `rgba(212, 76, 86, ${opacity})`,
    successAlpha: (opacity: number) => `rgba(52, 176, 125, ${opacity})`,
    whiteAlpha: (opacity: number) => `rgba(255, 255, 255, ${opacity})`,
  };
}
