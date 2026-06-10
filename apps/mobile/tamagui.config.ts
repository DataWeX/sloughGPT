import { createTamagui } from '@tamagui/core'
import { config as tamaguiConfig } from '@tamagui/config'

const appConfig = createTamagui(tamaguiConfig)

export default appConfig

export type AppConfig = typeof appConfig

declare module 'tamagui' {
  interface TamaguiCustomConfig extends AppConfig {}
}
