type Listener<T> = (payload: T) => void;

/** Minimal typed pub/sub, no external dependency. */
export class TypedEventEmitter<EventMap extends Record<string, unknown>> {
  private listeners: { [K in keyof EventMap]?: Set<Listener<EventMap[K]>> } = {};

  on<K extends keyof EventMap>(event: K, listener: Listener<EventMap[K]>): () => void {
    let set = this.listeners[event];
    if (!set) {
      set = new Set();
      this.listeners[event] = set;
    }
    set.add(listener);
    return () => this.off(event, listener);
  }

  off<K extends keyof EventMap>(event: K, listener: Listener<EventMap[K]>): void {
    this.listeners[event]?.delete(listener);
  }

  protected emit<K extends keyof EventMap>(event: K, payload: EventMap[K]): void {
    this.listeners[event]?.forEach((listener) => listener(payload));
  }
}
