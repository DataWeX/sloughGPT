import * as Sentry from '@sentry/react-native';

const SENTRY_DSN = process.env.EXPO_PUBLIC_SENTRY_DSN || 'https://7aa3997a8d94c4efad55d12d520d4023@o4511484757540864.ingest.de.sentry.io/4511484821373008';

// Only initialize Sentry if DSN is provided
if (SENTRY_DSN) {
  Sentry.init({
    dsn: SENTRY_DSN,
    
    // Set environment
    environment: __DEV__ ? 'development' : 'production',
    
    // Enable performance monitoring
    tracesSampleRate: __DEV__ ? 1.0 : 0.2,
    
    // Enable profiling
    profilesSampleRate: __DEV__ ? 1.0 : 0.1,
    
    // Ignore certain errors
    ignoreErrors: [
      'Network request failed',
      'AbortError',
      'TimeoutError',
    ],
    
    // Before send hook to sanitize data
    beforeSend(event) {
      if (event.request?.headers) {
        delete event.request.headers['Authorization'];
      }
      return event;
    },
    
    // Debug mode in development
    debug: __DEV__,
  });
}

export { Sentry };
