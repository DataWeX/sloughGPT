'use client'

import { createContext, useContext, useState, useCallback, ReactNode, useEffect } from 'react'

type Locale = 'en' | 'es' | 'fr' | 'de' | 'zh'

interface Translations {
  [key: string]: string
}

const translations: Record<Locale, Translations> = {
  en: {
    'app.name': 'Man',
    'app.console': 'Console',
    'common.starting': 'Starting...',
    'common.starting_sub': 'Model is loading, one moment',
    'common.save': 'Save',
    'nav.chat': 'Chat',
    'nav.models': 'Personalities',
    'nav.datasets': 'Datasets',
    'nav.training': 'Teach me',
    'nav.settings': 'Settings',
    'nav.multimodal': 'Multimodal',
    'nav.agents': 'Agents',
    'nav.knowledge': 'Knowledge',
    'nav.monitoring': 'Health',
    'nav.export': 'Export',
    'nav.adapters': 'Adapters',
    'nav.errors': 'Errors',
    'nav.experiments': 'Experiments',
    'nav.workflow': 'Workflow',
    'nav.feedback': 'Feedback',
    'nav.files': 'Files',
    'nav.souls': 'Souls',
    'nav.registry': 'Registry',
    'nav.security': 'Security',
    'nav.images': 'Images',
    'nav.auth': 'Auth',
    'nav.vm': 'VM',
    'nav.benchmark': 'Benchmark',
    'nav.compare': 'Compare',
    'nav.tokenizer': 'Tokenizer',
    'nav.learn': 'Learn',
    'nav.companion': 'Companion',
    'nav.voice': 'Voice',
    'nav.section.core': 'Core',
    'nav.section.ai': 'AI',
    'nav.section.system': 'System',
    'nav.section.tools': 'Tools',
    'home.subtitle.offline': 'Offline',
    'home.apiOffline.title': 'Service Offline',
    'home.apiOffline.body': 'Service at {url} is not reachable',
    'home.stats.status': 'Status',
    'home.stats.models': 'Models',
    'home.stats.personality': 'Personality',
    'chat.send': 'Send',
    'sidebar.close': 'Close menu',
    'sidebar.home': 'Home',
  },

  es: {
    'app.name': 'Man',
    'app.console': 'Consola',
    'common.starting': 'Iniciando...',
    'common.starting_sub': 'El modelo está cargando, un momento',
    'common.save': 'Guardar',
    'nav.chat': 'Chat',
    'nav.models': 'Personalidades',
    'nav.datasets': 'Datos',
    'nav.training': 'Enseñame',
    'nav.settings': 'Configuración',
    'nav.multimodal': 'Multimodal',
    'nav.agents': 'Agentes',
    'nav.knowledge': 'Conocimiento',
    'nav.monitoring': 'Salud',
    'nav.export': 'Exportar',
    'nav.adapters': 'Adaptadores',
    'nav.errors': 'Errores',
    'nav.experiments': 'Experimentos',
    'nav.workflow': 'Flujo',
    'nav.feedback': 'Comentarios',
    'nav.files': 'Archivos',
    'nav.souls': 'Almas',
    'nav.registry': 'Registro',
    'nav.security': 'Seguridad',
    'nav.images': 'Imágenes',
    'nav.auth': 'Autenticación',
    'nav.vm': 'VM',
    'nav.benchmark': 'Benchmark',
    'nav.compare': 'Comparar',
    'nav.tokenizer': 'Tokenizador',
    'nav.learn': 'Aprender',
    'nav.companion': 'Compañero',
    'nav.voice': 'Voz',
    'nav.section.core': 'Principal',
    'nav.section.ai': 'IA',
    'nav.section.system': 'Sistema',
    'nav.section.tools': 'Herramientas',
    'home.subtitle.offline': 'Sin conexión',
    'home.apiOffline.title': 'Servicio no accesible',
    'home.stats.status': 'Estado',
    'home.stats.models': 'Modelos',
    'home.stats.personality': 'Personalidad',
    'chat.send': 'Enviar',
    'sidebar.close': 'Cerrar menú',
    'sidebar.home': 'Inicio',
    'home.apiOffline.body': 'El servidor API en {url} no es accesible',
  },

  fr: {
    'app.name': 'Man',
    'app.console': 'Console',
    'common.starting': 'Démarrage...',
    'common.starting_sub': 'Le modèle se charge, un instant',
    'common.save': 'Enregistrer',
    'nav.chat': 'Chat',
    'nav.models': 'Personnalités',
    'nav.datasets': 'Données',
    'nav.training': 'Enseigne-moi',
    'nav.settings': 'Paramètres',
    'nav.multimodal': 'Multimodal',
    'nav.agents': 'Agents',
    'nav.knowledge': 'Connaissances',
    'nav.monitoring': 'Santé',
    'nav.export': 'Exporter',
    'nav.adapters': 'Adaptateurs',
    'nav.errors': 'Erreurs',
    'nav.experiments': 'Expériences',
    'nav.workflow': 'Flux',
    'nav.feedback': 'Retour',
    'nav.files': 'Fichiers',
    'nav.souls': 'Âmes',
    'nav.registry': 'Registre',
    'nav.security': 'Sécurité',
    'nav.images': 'Images',
    'nav.auth': 'Authentification',
    'nav.vm': 'VM',
    'nav.benchmark': 'Benchmark',
    'nav.compare': 'Comparer',
    'nav.tokenizer': 'Tokenizer',
    'nav.learn': 'Apprendre',
    'nav.companion': 'Compagnon',
    'nav.voice': 'Voix',
    'nav.section.core': 'Principal',
    'nav.section.ai': 'IA',
    'nav.section.system': 'Système',
    'nav.section.tools': 'Outils',
    'home.subtitle.offline': 'Hors ligne',
    'home.apiOffline.title': 'Service inaccessible',
    'home.stats.status': 'Statut',
    'home.stats.models': 'Modèles',
    'home.stats.personality': 'Personnalité',
    'chat.send': 'Envoyer',
    'sidebar.close': 'Fermer le menu',
    'sidebar.home': 'Accueil',
    'home.apiOffline.body': 'Le serveur API à {url} est inaccessible',
  },

  de: {
    'app.name': 'Man',
    'app.console': 'Konsole',
    'common.starting': 'Starte...',
    'common.starting_sub': 'Modell wird geladen, einen Moment',
    'common.save': 'Speichern',
    'nav.chat': 'Chat',
    'nav.models': 'Persönlichkeiten',
    'nav.datasets': 'Datensätze',
    'nav.training': 'Bring mir bei',
    'nav.settings': 'Einstellungen',
    'nav.multimodal': 'Multimodal',
    'nav.agents': 'Agenten',
    'nav.knowledge': 'Wissen',
    'nav.monitoring': 'Gesundheit',
    'nav.export': 'Exportieren',
    'nav.adapters': 'Adapter',
    'nav.errors': 'Fehler',
    'nav.experiments': 'Experimente',
    'nav.workflow': 'Workflow',
    'nav.feedback': 'Feedback',
    'nav.files': 'Dateien',
    'nav.souls': 'Seelen',
    'nav.registry': 'Registrierung',
    'nav.security': 'Sicherheit',
    'nav.images': 'Bilder',
    'nav.auth': 'Authentifizierung',
    'nav.vm': 'VM',
    'nav.benchmark': 'Benchmark',
    'nav.compare': 'Vergleichen',
    'nav.tokenizer': 'Tokenizer',
    'nav.learn': 'Lernen',
    'nav.companion': 'Begleiter',
    'nav.voice': 'Stimme',
    'nav.section.core': 'Kern',
    'nav.section.ai': 'KI',
    'nav.section.system': 'System',
    'nav.section.tools': 'Werkzeuge',
    'home.subtitle.offline': 'Offline',
    'home.apiOffline.title': 'Dienst nicht erreichbar',
    'home.stats.status': 'Status',
    'home.stats.models': 'Modelle',
    'home.stats.personality': 'Persönlichkeit',
    'chat.send': 'Senden',
    'sidebar.close': 'Menü schließen',
    'sidebar.home': 'Startseite',
    'home.apiOffline.body': 'Dienst unter {url} ist nicht erreichbar',
  },

  zh: {
    'app.name': 'Man',
    'app.console': '控制台',
    'common.starting': '启动中...',
    'common.starting_sub': '模型正在加载，请稍候',
    'common.save': '保存',
    'nav.chat': '聊天',
    'nav.models': '个性',
    'nav.datasets': '数据集',
    'nav.training': '教我',
    'nav.settings': '设置',
    'nav.multimodal': '多模态',
    'nav.agents': '智能体',
    'nav.knowledge': '知识库',
    'nav.monitoring': '健康',
    'nav.export': '导出',
    'nav.adapters': '适配器',
    'nav.errors': '错误',
    'nav.experiments': '实验',
    'nav.workflow': '工作流',
    'nav.feedback': '反馈',
    'nav.files': '文件',
    'nav.souls': '灵魂',
    'nav.registry': '注册表',
    'nav.security': '安全',
    'nav.images': '图片',
    'nav.auth': '认证',
    'nav.vm': 'VM',
    'nav.benchmark': '基准测试',
    'nav.compare': '比较',
    'nav.tokenizer': '分词器',
    'nav.learn': '学习',
    'nav.companion': '伙伴',
    'nav.voice': '语音',
    'nav.section.core': '核心',
    'nav.section.ai': 'AI',
    'nav.section.system': '系统',
    'nav.section.tools': '工具',
    'home.subtitle.offline': '离线',
    'home.apiOffline.title': 'Service Unavailable',
    'home.stats.status': '状态',
    'home.stats.models': '模型',
    'home.stats.personality': '个性',
    'chat.send': '发送',
    'sidebar.close': '关闭菜单',
    'sidebar.home': '首页',
    'home.apiOffline.body': 'Service at {url} is not reachable',
  },
}

interface LocaleContextType {
  locale: Locale
  setLocale: (locale: Locale) => void
  t: (key: string, params?: Record<string, string | number>) => string
  locales: Locale[]
}

const LocaleContext = createContext<LocaleContextType | undefined>(undefined)

const LOCALE_KEY = 'man_locale'

export function LocaleProvider({ children }: { children: ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>(() => {
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem(LOCALE_KEY) as Locale
      if (saved && translations[saved]) return saved
    }
    return 'en'
  })

  const setLocale = useCallback((newLocale: Locale) => {
    setLocaleState(newLocale)
    if (typeof window !== 'undefined') {
      localStorage.setItem(LOCALE_KEY, newLocale)
      document.documentElement.lang = newLocale
    }
  }, [])

  useEffect(() => {
    document.documentElement.lang = locale
  }, [locale])

  const t = useCallback((key: string, params?: Record<string, string | number>): string => {
    let text = translations[locale]?.[key] || translations.en?.[key] || key
    if (params) {
      for (const [k, v] of Object.entries(params)) {
        text = text.replace(`{${k}}`, String(v))
      }
    }
    return text
  }, [locale])

  return (
    <LocaleContext.Provider value={{ locale, setLocale, t, locales: Object.keys(translations) as Locale[] }}>
      {children}
    </LocaleContext.Provider>
  )
}

export function useLocale() {
  const context = useContext(LocaleContext)
  if (!context) {
    throw new Error('useLocale must be used within a LocaleProvider')
  }
  return context
}

export const LOCALES = [
  { code: 'en', name: 'English', flag: '🇺🇸' },
  { code: 'es', name: 'Español', flag: '🇪🇸' },
  { code: 'fr', name: 'Français', flag: '🇫🇷' },
  { code: 'de', name: 'Deutsch', flag: '🇩🇪' },
  { code: 'zh', name: '中文', flag: '🇨🇳' },
] as const
