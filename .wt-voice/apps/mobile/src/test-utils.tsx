import React, {type ReactElement} from 'react'
import {TamaguiProvider, type TamaguiProviderProps} from 'tamagui'
import {render, type RenderOptions} from '@testing-library/react-native'
import appConfig from '../tamagui.config'

function AllTheProviders({children, ...props}: Partial<TamaguiProviderProps>) {
  return (
    <TamaguiProvider config={appConfig} defaultTheme="light" {...props}>
      {children}
    </TamaguiProvider>
  )
}

function customRender(ui: ReactElement, options?: Omit<RenderOptions, 'wrapper'>) {
  return render(ui, {wrapper: AllTheProviders, ...options})
}

// Re-export everything
export * from '@testing-library/react-native'
export {customRender as render}
