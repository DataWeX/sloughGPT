# ProGuard rules for SloughGPT

# Keep app classes
-keep class com.sloughgpt.** { *; }

# React Native
-keep,allowobfuscation @interface com.facebook.proguard.annotations.DoNotStrip
-keep,allowobfuscation @interface com.facebook.proguard.annotations.KeepGettersAndSetters
-keep @com.facebook.proguard.annotations.DoNotStrip class *
-keepclassmembers class * {
    @com.facebook.proguard.annotations.DoNotStrip *;
    @com.facebook.proguard.annotations.KeepGettersAndSetters *;
}
-keepclassmembers @com.facebook.proguard.annotations.KeepGettersAndSetters class * {
    void set*(***);
    *** get*();
}

# Hermes
-keep class com.facebook.hermes.unicode.** { *; }
-keep class com.facebook.jni.** { *; }

# React Native SVG
-keep public class com.horcrux.svg.** { *; }

# Expo Modules
-keep class expo.modules.** { *; }

# Navigation
-keep class com.facebook.react.turbomodule.** { *; }

# Don't strip native methods
-keepclasseswithmembernames class * {
    native <methods>;
}

# Keep Jackson/JSON serialization
-keepattributes Signature
-keepattributes *Annotation*
-keep class sun.misc.Unsafe { *; }
-keep class com.google.gson.** { *; }
-keep class org.json.** { *; }

# OkHttp / httpx
-dontwarn okhttp3.**
-dontwarn okio.**
-dontwarn javax.annotation.**
-dontwarn org.conscrypt.**
