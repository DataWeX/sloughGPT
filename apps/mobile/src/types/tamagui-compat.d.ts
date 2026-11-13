/**
 * Ambient declarations for Tamagui compatibility in React Native.
 *
 * @tamagui/element references HTMLElement as a value in its web entrypoint.
 * React Native's TS environment does not include lib.dom.d.ts, so HTMLElement
 * is only a type (from global declarations), not a constructor value. This
 * minimal ambient declaration satisfies the type-checker without polluting the
 * RN global scope with DOM APIs.
 */
declare class HTMLElement {}
