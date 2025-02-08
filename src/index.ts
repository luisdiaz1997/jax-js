import * as core from "./core";
import * as numpy from "./numpy";
import { Array, ArrayLike } from "./numpy";
import * as tree from "./tree";
import type { JsTree } from "./tree";

export { numpy, tree };

// Convert a subtype of JsTree<A> into a JsTree<B>, with the same structure.
type MapJsTree<T, A, B> = T extends A
  ? B
  : T extends globalThis.Array<infer U>
    ? MapJsTree<U, A, B>[]
    : { [K in keyof T]: MapJsTree<T[K], A, B> };

// Assert that a function's arguments are a subtype of the given type.
type WithArgsSubtype<F extends (args: any[]) => any, T> =
  Parameters<F> extends T ? F : never;

/** Compute the forward-mode Jacobian-vector product for a function. */
export const jvp = core.jvp as <F extends (...args: any[]) => JsTree<Array>>(
  f: WithArgsSubtype<F, JsTree<ArrayLike>>,
  primals: MapJsTree<Parameters<F>, Array, ArrayLike>,
  tangents: MapJsTree<Parameters<F>, Array, ArrayLike>
) => [ReturnType<F>, ReturnType<F>];

/** Vectorize an operation on a batched axis for one or more inputs. */
export const vmap = core.vmap as <F extends (...args: any[]) => JsTree<Array>>(
  f: WithArgsSubtype<F, JsTree<ArrayLike>>,
  inAxes: MapJsTree<Parameters<F>, Array, number>
) => F;

/** Compute the Jacobian evaluated column-by-column by forward-mode AD. */
export const jacfwd = core.jacfwd as <F extends (x: Array) => Array>(
  f: F,
  x: Array
) => F;
