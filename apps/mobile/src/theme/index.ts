import {Dimensions, Platform} from 'react-native';

const {width: SCREEN_WIDTH} = Dimensions.get('window');

export const colors = {
  primary: '#7C52C4',
  primaryLight: '#C0AAF4',
  accent: '#EC915F',
  success: '#34B07D',
  warning: '#ECA83C',
  error: '#DC505A',
  background: '#FFFFFF',
  surface: '#F5F3F7',
  surfaceHover: '#EDEBF2',
  border: '#E0DCE8',
  text: '#1A1625',
  textSecondary: '#6B6580',
  textMuted: '#9B95A8',
  white: '#FFFFFF',
  black: '#000000',
};

export const darkColors = {
  ...colors,
  background: '#0F0D15',
  surface: '#1A1725',
  surfaceHover: '#252235',
  border: '#2D2A3A',
  text: '#F0ECF5',
  textSecondary: '#9B95A8',
  textMuted: '#6B6580',
};

export const spacing = {
  xs: 4,
  sm: 8,
  md: 12,
  lg: 16,
  xl: 20,
  xxl: 24,
  xxxl: 32,
};

export const radii = {
  sm: 6,
  md: 10,
  lg: 16,
  xl: 20,
  full: 9999,
};

export const typography = {
  h1: {fontSize: 28, fontWeight: '700' as const, letterSpacing: -0.5},
  h2: {fontSize: 22, fontWeight: '600' as const, letterSpacing: -0.3},
  h3: {fontSize: 18, fontWeight: '600' as const},
  body: {fontSize: 15, fontWeight: '400' as const, lineHeight: 22},
  caption: {fontSize: 13, fontWeight: '400' as const, lineHeight: 18},
  small: {fontSize: 11, fontWeight: '500' as const, letterSpacing: 0.3},
  mono: {
    fontSize: 13,
    fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
  },
};

export const layout = {
  screenPadding: spacing.lg,
  touchTarget: 44,
  maxContentWidth: SCREEN_WIDTH - spacing.lg * 2,
};
