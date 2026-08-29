/**
 * IndexedDB-backed weight cache for .sou checkpoints.
 *
 * Avoids re-fetching and re-parsing large weight files across page reloads.
 * Keys by URL; gracefully degrades when IndexedDB is unavailable.
 */

const DB_NAME = 'soulnet-weights'
const STORE = 'checkpoints'
const DB_VERSION = 1

function idbRequest<T>(req: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result)
    req.onerror = () => reject(req.error)
  })
}

function idbTxComplete(tx: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    tx.oncomplete = () => resolve()
    tx.onerror = () => reject(tx.error)
    tx.onabort = () => reject(tx.error)
  })
}

export class WeightCache {
  private dbPromise: Promise<IDBDatabase> | null = null
  private openAttempted = false

  private getDB(): Promise<IDBDatabase> | null {
    if (typeof indexedDB === 'undefined') return null
    if (!this.dbPromise) {
      this.dbPromise = new Promise<IDBDatabase>((resolve, reject) => {
        const req = indexedDB.open(DB_NAME, DB_VERSION)
        req.onupgradeneeded = () => {
          req.result.createObjectStore(STORE)
        }
        req.onsuccess = () => {
          this.openAttempted = false
          resolve(req.result)
        }
        req.onerror = () => {
          this.openAttempted = true
          this.dbPromise = null
          reject(req.error)
        }
      })
    }
    return this.dbPromise
  }

  async get(url: string): Promise<ArrayBuffer | null> {
    try {
      const db = await this.getDB()
      if (!db) return null
      const tx = db.transaction(STORE, 'readonly')
      const store = tx.objectStore(STORE)
      const result = await idbRequest<ArrayBuffer | undefined>(store.get(url))
      return result ?? null
    } catch {
      return null
    }
  }

  async put(url: string, buffer: ArrayBuffer): Promise<void> {
    try {
      const db = await this.getDB()
      if (!db) return
      const tx = db.transaction(STORE, 'readwrite')
      const store = tx.objectStore(STORE)
      store.put(buffer, url)
      await idbTxComplete(tx)
    } catch {
      // silently ignore write failures
    }
  }

  async delete(url: string): Promise<void> {
    try {
      const db = await this.getDB()
      if (!db) return
      const tx = db.transaction(STORE, 'readwrite')
      const store = tx.objectStore(STORE)
      store.delete(url)
      await idbTxComplete(tx)
    } catch {
      // silently ignore delete failures
    }
  }

  async clear(): Promise<void> {
    try {
      const db = await this.getDB()
      if (!db) return
      const tx = db.transaction(STORE, 'readwrite')
      const store = tx.objectStore(STORE)
      store.clear()
      await idbTxComplete(tx)
    } catch {
      // silently ignore clear failures
    }
  }
}
