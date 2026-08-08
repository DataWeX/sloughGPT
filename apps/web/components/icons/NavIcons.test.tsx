import { describe, it, expect } from 'vitest'

import * as NavIcons from './NavIcons'

const EXPECTED_EXPORTS = [
  'IconChat',
  'IconModels',
  'IconSettings',
  'IconMoon',
  'IconSun',
  'IconMenu',
  'IconActivity',
  'IconCompare',
  'IconX',
  'IconSearch',
  'IconPlus',
  'IconTrash',
  'IconDownload',
  'IconAlert',
  'IconUpload',
  'IconTraining',
  'IconBenchmark',
  'IconTokenizer',
  'IconExport',
  'IconLabs',
  'IconAgents',
  'IconLogin',
  'IconBrain',
  'IconVision',
  'IconFile',
  'IconChangelog',
  'IconClock',
  'IconClose',
]

describe('NavIcons barrel', () => {
  it('exports every icon name', () => {
    for (const name of EXPECTED_EXPORTS) {
      expect(NavIcons, `missing export ${name}`).toHaveProperty(name)
    }
  })

  it('every export is a callable component', () => {
    for (const name of EXPECTED_EXPORTS) {
      expect(typeof (NavIcons as Record<string, unknown>)[name]).toBe('function')
    }
  })

  it('IconClose is an alias of IconX', () => {
    expect(NavIcons.IconClose).toBe(NavIcons.IconX)
  })
})
