/**
 * Minimal browser EventEmitter for the single transitive Node-builtin edge in
 * the admitted dependency tree (xmlbuilder2 <- @kitware/vtk.js IO/XML).
 *
 * @kitware/vtk.js's static IO barrel pulls in xmlbuilder2, whose CommonJS
 * bundle does `class XMLBuilderCBImpl extends require("events").EventEmitter`
 * at module scope.  A browser bundler has no Node `events` builtin, so without
 * this shim the module throws at bundle evaluation even though the CCTA
 * physician-viewer never exercises XML IO.  The shim keeps that module
 * constructible; the XML builder API itself stays untouched (calling it would
 * fail in xmlbuilder2 on its own terms if the optional XML helper is absent).
 *
 * This is NOT a dependency: it is a local, dependency-free compatibility
 * shim inside webapp/ (no registry package is added; the pins stay frozen).
 */
export class EventEmitter {
  constructor() {
    this._listeners = new Map();
  }

  on(event, listener) {
    if (!this._listeners.has(event)) this._listeners.set(event, []);
    this._listeners.get(event).push(listener);
    return this;
  }

  addListener(event, listener) {
    return this.on(event, listener);
  }

  once(event, listener) {
    const wrapper = (...args) => {
      this.off(event, wrapper);
      listener.apply(this, args);
    };
    return this.on(event, wrapper);
  }

  prependListener(event, listener) {
    if (!this._listeners.has(event)) this._listeners.set(event, []);
    this._listeners.get(event).unshift(listener);
    return this;
  }

  off(event, listener) {
    const listeners = this._listeners.get(event);
    if (listeners) {
      this._listeners.set(
        event,
        listeners.filter((entry) => entry !== listener)
      );
    }
    return this;
  }

  removeListener(event, listener) {
    return this.off(event, listener);
  }

  removeAllListeners(event) {
    if (event === undefined) this._listeners.clear();
    else this._listeners.delete(event);
    return this;
  }

  listeners(event) {
    return [...(this._listeners.get(event) || [])];
  }

  listenerCount(event) {
    return (this._listeners.get(event) || []).length;
  }

  eventNames() {
    return [...this._listeners.keys()];
  }

  emit(event, ...args) {
    const listeners = [...(this._listeners.get(event) || [])];
    for (const listener of listeners) listener.apply(this, args);
    return listeners.length > 0;
  }
}

export default { EventEmitter };
