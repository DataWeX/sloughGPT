export class InterceptorManager<T> {
  private _interceptors: Array<{
    id: number
    fulfilled: (value: T) => T | Promise<T>
    rejected?: (error: unknown) => unknown
  }> = []
  private _nextId = 0

  use(fulfilled: (value: T) => T | Promise<T>, rejected?: (error: unknown) => unknown): number {
    const id = this._nextId++
    this._interceptors.push({ id, fulfilled, rejected })
    return id
  }

  eject(id: number) {
    this._interceptors = this._interceptors.filter((i) => i.id !== id)
  }

  clear() {
    this._interceptors = []
  }

  get size(): number {
    return this._interceptors.length
  }

  async run(initial: T): Promise<T> {
    let value = initial
    for (const interceptor of this._interceptors) {
      try {
        value = await interceptor.fulfilled(value)
      } catch (e) {
        if (interceptor.rejected) {
          value = (await interceptor.rejected(e)) as T
        } else {
          throw e
        }
      }
    }
    return value
  }
}
