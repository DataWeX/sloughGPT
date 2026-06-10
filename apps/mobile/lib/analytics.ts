import * as Sentry from '@sentry/react-native';

/**
 * Performance tracking utilities for the mobile app
 */
export const PerformanceTracker = {
  /**
   * Start a performance span
   */
  startSpan(name: string, op: string) {
    return Sentry.startInactiveSpan({
      name,
      op,
    });
  },

  /**
   * Track a custom span within a transaction
   */
  async trackSpan<T>(
    name: string,
    op: string,
    fn: () => Promise<T>
  ): Promise<T> {
    return Sentry.startSpan(
      {
        name,
        op,
      },
      async () => {
        return await fn();
      }
    );
  },

  /**
   * Track API request performance
   */
  async trackApiRequest<T>(
    endpoint: string,
    method: string,
    fn: () => Promise<T>
  ): Promise<T> {
    return Sentry.startSpan(
      {
        name: `${method} ${endpoint}`,
        op: 'http.client',
      },
      async () => {
        try {
          const result = await fn();
          return result;
        } catch (error) {
          Sentry.captureException(error);
          throw error;
        }
      }
    );
  },

  /**
   * Track screen load performance
   */
  trackScreenLoad(screenName: string) {
    const span = Sentry.startInactiveSpan({
      name: `Screen: ${screenName}`,
      op: 'ui.load',
    });

    return {
      finish: () => {
        span?.end();
      },
      cancel: () => {
        span?.end();
      },
    };
  },

  /**
   * Track user interaction
   */
  trackInteraction(action: string, component: string, metadata?: Record<string, any>) {
    Sentry.addBreadcrumb({
      category: 'ui.click',
      message: `${action} on ${component}`,
      level: 'info',
      data: metadata,
    });
  },

  /**
   * Track navigation
   */
  trackNavigation(from: string, to: string) {
    Sentry.addBreadcrumb({
      category: 'navigation',
      message: `Navigated from ${from} to ${to}`,
      level: 'info',
    });
  },

  /**
   * Track error with context
   */
  trackError(error: Error, context?: Record<string, any>) {
    Sentry.captureException(error, {
      extra: context,
    });
  },

  /**
   * Set user context for error tracking
   */
  setUser(userId: string, username?: string, email?: string) {
    Sentry.setUser({
      id: userId,
      username,
      email,
    });
  },

  /**
   * Clear user context
   */
  clearUser() {
    Sentry.setUser(null);
  },

  /**
   * Track custom metric
   */
  trackMetric(name: string, value: number, unit: string = 'none') {
    // Sentry doesn't have direct metrics API in SDK, use tags instead
    Sentry.setTag(`metric_${name}`, `${value}_${unit}`);
  },
};

/**
 * Analytics tracking for user events
 */
export const Analytics = {
  /**
   * Track a custom event
   */
  trackEvent(event: string, properties?: Record<string, any>) {
    Sentry.addBreadcrumb({
      category: 'custom',
      message: event,
      level: 'info',
      data: properties,
    });

    // Also send as a custom event for analytics
    Sentry.captureMessage(event, {
      level: 'info',
      extra: properties,
    });
  },

  /**
   * Track feature usage
   */
  trackFeatureUsage(feature: string, action: string, metadata?: Record<string, any>) {
    this.trackEvent(`feature_${feature}_${action}`, metadata);
  },

  /**
   * Track chat message sent
   */
  trackChatMessageSent(metadata?: { hasImages?: boolean; messageLength?: number }) {
    this.trackEvent('chat_message_sent', metadata);
  },

  /**
   * Track model switch
   */
  trackModelSwitch(modelId: string) {
    this.trackEvent('model_switched', { modelId });
  },

  /**
   * Track soul switch
   */
  trackSoulSwitch(soulName: string) {
    this.trackEvent('soul_switched', { soulName });
  },

  /**
   * Track knowledge item added
   */
  trackKnowledgeAdded(metadata?: { topic?: string; source?: string }) {
    this.trackEvent('knowledge_added', metadata);
  },

  /**
   * Track session created
   */
  trackSessionCreated() {
    this.trackEvent('session_created');
  },

  /**
   * Track settings changed
   */
  trackSettingsChanged(setting: string, value: any) {
    this.trackEvent('settings_changed', { setting, value });
  },
};
