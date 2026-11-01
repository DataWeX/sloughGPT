'use client'

import { render, screen } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import ConsciousnessHelpPage from './page'

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn() }),
}))

vi.mock('@/hooks/useLocale', () => ({
  useLocale: () => ({
    t: (key: string) => {
      const translations: Record<string, string> = {
        'consciousness_help.page_title': 'Consciousness Help',
        'consciousness_help.faq_title': 'FAQ',
        'consciousness_help.faq_desc': 'Frequently asked questions',
        'consciousness_help.quick_links_title': 'Quick Links',
        'consciousness_help.quick_links_desc': 'Navigate to consciousness pages',
        'consciousness_help.shortcuts_title': 'Keyboard Shortcuts',
        'consciousness_help.shortcuts_desc': 'Quick access shortcuts',
        'consciousness_help.system_info_title': 'System Info',
        'consciousness_help.system_info_desc': 'Current system information',
        'consciousness_help.version': 'Version',
        'consciousness_help.api_endpoint': 'API Endpoint',
        'consciousness_help.total_pages': 'Total Pages',
        'consciousness_help.total_endpoints': 'Total Endpoints',
      }
      return translations[key] || key
    },
  }),
}))

describe('ConsciousnessHelpPage', () => {
  it('renders without crashing', () => {
    render(<ConsciousnessHelpPage />)
    expect(screen.getByText('Consciousness Help')).toBeTruthy()
  })

  it('shows FAQ section', () => {
    render(<ConsciousnessHelpPage />)
    expect(screen.getByText('FAQ')).toBeTruthy()
  })

  it('shows quick links section', () => {
    render(<ConsciousnessHelpPage />)
    expect(screen.getByText('Quick Links')).toBeTruthy()
  })

  it('shows keyboard shortcuts section', () => {
    render(<ConsciousnessHelpPage />)
    expect(screen.getByText('Keyboard Shortcuts')).toBeTruthy()
  })

  it('shows system info section', () => {
    render(<ConsciousnessHelpPage />)
    expect(screen.getByText('System Info')).toBeTruthy()
  })

  it('shows version info', () => {
    render(<ConsciousnessHelpPage />)
    expect(screen.getByText('Version')).toBeTruthy()
    expect(screen.getByText('3.0.0')).toBeTruthy()
  })

  it('shows total pages count', () => {
    render(<ConsciousnessHelpPage />)
    expect(screen.getByText('Total Pages')).toBeTruthy()
    expect(screen.getByText('26')).toBeTruthy()
  })

  it('shows total endpoints count', () => {
    render(<ConsciousnessHelpPage />)
    expect(screen.getByText('Total Endpoints')).toBeTruthy()
    expect(screen.getByText('37')).toBeTruthy()
  })
})
