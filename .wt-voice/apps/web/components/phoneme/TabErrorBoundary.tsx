'use client'

import { Component, type ReactNode } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button } from '@sloughgpt/strui'

interface Props {
  children: ReactNode
  tabName: string
}

interface State {
  hasError: boolean
  error: Error | null
}

export default class TabErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false, error: null }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null })
  }

  render() {
    if (this.state.hasError) {
      return (
        <Card>
          <CardHeader>
            <CardTitle className="text-destructive">Something went wrong</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <p className="text-sm text-muted-foreground">
              The {this.props.tabName} tab encountered an error.
            </p>
            {this.state.error && (
              <code className="block text-xs p-2 rounded bg-background font-mono text-destructive">
                {this.state.error.message}
              </code>
            )}
            <Button variant="secondary" size="sm" onClick={this.handleReset}>
              Try again
            </Button>
          </CardContent>
        </Card>
      )
    }

    return this.props.children
  }
}
