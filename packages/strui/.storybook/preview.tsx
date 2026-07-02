import type { Preview } from '@storybook/react'
import type { ReactNode } from 'react'
import { useEffect } from 'react'

import { cn } from '../src/lib/cn'
import '../src/styles/globals.css'
import './preview.css'

function ThemeSync({
  theme,
  children,
}: {
  theme: string
  children: ReactNode
}) {
  useEffect(() => {
    const root = document.documentElement
    root.classList.toggle('dark', theme === 'dark')
    return () => root.classList.remove('dark')
  }, [theme])
  return <>{children}</>
}

const preview: Preview = {
  globalTypes: {
    theme: {
      description: 'Pastel lattice — light or dark surface',
      defaultValue: 'light',
      toolbar: {
        title: 'Surface',
        icon: 'mirror',
        items: [
          { value: 'light', title: 'Light', icon: 'sun' },
          { value: 'dark', title: 'Dark', icon: 'moon' },
        ],
        dynamicTitle: true,
      },
    },
  },
  initialGlobals: {
    theme: 'light',
  },
  parameters: {
    layout: 'centered',
    backgrounds: { disable: true },
    controls: {
      sort: 'requiredFirst',
      expanded: true,
      matchers: {
        color: /(background|color)$/i,
        date: /Date$/,
      },
    },
    docs: {
      toc: true,
      canvas: { sourceState: 'shown' },
    },
    options: {
      storySort: {
        order: [
          'Docs',
          ['Introduction', 'Design principles', 'Foundations', 'Component gallery'],
          'UI',
          'Composed',
          'AI',
        ],
      },
    },
  },
  decorators: [
    (Story, context) => {
      const theme = (context.globals.theme as string) ?? 'light'
      const fullscreen = context.parameters.layout === 'fullscreen'
      return (
        <ThemeSync theme={theme}>
          <div
            className={cn(
              'sb-strui-root text-foreground antialiased',
              fullscreen && 'sb-strui-root--fullscreen',
            )}
          >
            <Story />
          </div>
        </ThemeSync>
      )
    },
  ],
}

export default preview
