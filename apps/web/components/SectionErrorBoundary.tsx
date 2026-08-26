'use client'

import { Component, ReactNode, useCallback, useState } from 'react'
import { Card, CardContent, Button } from '@sloughgpt/strui'

interface Props {
  children: ReactNode
  sectionName?: string
}

interface State {
  hasError: boolean
  error?: Error
}

export class SectionErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = { hasError: false }
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, errorInfo: { componentStack?: string }) {
    console.error(`SectionErrorBoundary (${this.props.sectionName || 'section'}) caught:`, error, errorInfo)
  }

  reset() {
    this.setState({ hasError: false, error: undefined })
  }

  render() {
    if (this.state.hasError) {
      return (
        <Card className="border-destructive/30">
          <CardContent className="p-4 text-center space-y-2">
            <p className="text-sm text-destructive font-medium">Section failed to load</p>
            <p className="text-xs text-muted-foreground truncate max-w-xs mx-auto" title={this.state.error?.message}>
              {this.state.error?.message || 'An unexpected error occurred'}
            </p>
            <Button size="sm" variant="outline" onClick={() => this.reset()}>
              Retry
            </Button>
          </CardContent>
        </Card>
      )
    }
    return this.props.children
  }
}
