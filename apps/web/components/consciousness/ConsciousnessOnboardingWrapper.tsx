'use client'

import { useState, useEffect } from 'react'
import { ConsciousnessOnboarding } from './ConsciousnessOnboarding'

const ONBOARDING_KEY = 'consciousness_onboarding_complete'

function isOnboardingComplete(): boolean {
  if (typeof window === 'undefined') return true
  try {
    return localStorage.getItem(ONBOARDING_KEY) === 'true'
  } catch {
    return true
  }
}

export function ConsciousnessOnboardingWrapper({ children }: { children: React.ReactNode }) {
  const [show, setShow] = useState(false)

  useEffect(() => {
    if (!isOnboardingComplete()) {
      setShow(true)
    }
  }, [])

  const handleComplete = () => {
    try {
      localStorage.setItem(ONBOARDING_KEY, 'true')
    } catch {}
    setShow(false)
  }

  const handleDismiss = () => {
    try {
      localStorage.setItem(ONBOARDING_KEY, 'true')
    } catch {}
    setShow(false)
  }

  return (
    <>
      {children}
      {show && (
        <ConsciousnessOnboarding
          onComplete={handleComplete}
          onDismiss={handleDismiss}
        />
      )}
    </>
  )
}
