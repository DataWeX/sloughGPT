import { config } from '@tamagui/config'
import { createTamagui } from 'tamagui'

import { TAMAGUI_THEMES } from './src/theme/tamagui-themes.generated'

const appConfig = createTamagui({
  ...config,
  fonts: {
    ...config.fonts,
    body: {
      ...config.fonts.body,
      family: 'Outfit',
      weight: {
        1: '400',
        2: '400',
        3: '400',
        4: '400',
        5: '500',
        6: '500',
        7: '600',
        8: '600',
        9: '700',
        10: '700',
        11: '400',
        12: '400',
        13: '500',
        14: '500',
        15: '600',
        16: '700',
        true: '400',
      },
    },
    heading: {
      ...config.fonts.heading,
      family: 'Outfit',
      weight: {
        1: '400',
        2: '400',
        3: '500',
        4: '500',
        5: '600',
        6: '600',
        7: '700',
        8: '700',
        9: '700',
        10: '700',
        11: '400',
        12: '500',
        13: '600',
        14: '600',
        15: '700',
        16: '700',
        true: '400',
      },
    },
    mono: {
      ...config.fonts.mono,
      family: 'JetBrainsMono-Regular',
    },
  },
  themes: {
    ...config.themes,
    light: { ...config.themes.light, ...TAMAGUI_THEMES.light },
    dark: { ...config.themes.dark, ...TAMAGUI_THEMES.dark },
    purple: { ...config.themes.light, ...TAMAGUI_THEMES.purple },
    pink: { ...config.themes.light, ...TAMAGUI_THEMES.pink },
    ocean: { ...config.themes.light, ...TAMAGUI_THEMES.ocean },
    sunset: { ...config.themes.light, ...TAMAGUI_THEMES.sunset },
    mint: { ...config.themes.light, ...TAMAGUI_THEMES.mint },
  },
})

export type AppConfig = typeof appConfig

declare module 'tamagui' {
  interface TamaguiCustomConfig extends AppConfig {}
}

export default appConfig
