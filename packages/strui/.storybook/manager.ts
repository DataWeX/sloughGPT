import { addons } from '@storybook/manager-api'
import { create } from '@storybook/theming/create'

const struiTheme = create({
  base: 'light',
  brandTitle: 'strui',
  brandUrl: 'https://github.com/iamtowbee/sloughGPT',
  brandTarget: '_blank',

  colorPrimary: '#8b7bc4',
  colorSecondary: '#5eb89a',

  appBg: '#ede8f7',
  appContentBg: '#f3f0fa',
  appBorderColor: '#d4cae8',

  barBg: '#e8e2f4',
  barTextColor: '#3d3650',
  barSelectedColor: '#6b5aa8',
  barHoverColor: '#5c5468',

  inputBg: '#ffffff',
  inputBorder: '#cfc6e2',
  inputTextColor: '#262130',

  textColor: '#262130',
  textInverseColor: '#faf8ff',

  fontBase: '"Outfit", system-ui, sans-serif',
  fontCode: '"JetBrains Mono", ui-monospace, monospace',
})

addons.setConfig({
  theme: struiTheme,
  sidebar: {
    showRoots: true,
  },
})
