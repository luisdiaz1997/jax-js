/**
 * @file Lazy shape tracking for multidimensional tensors.
 *
 * This module provides an immutable `View` class that can be used to calculate
 * shapes of arrays as operations are applied to them, lazily.
 *
 * Some operations like reshape() may not be representable with a single view,
 * for instance, because composing reshape() with shrink() leads to a
 * non-contiguous range of memory locations. This is why `ShapeTracker` is a
 * list of views.
 *
 * Indexing into a `ShapeTracker` or `View` can be folded into shader code.
 *
 * This file is originally based on tinygrad's implementation of shape tracking
 * in the `tinygrad.shape` module.
 */

import { deepEqual, rep, zip } from "./utils";

/** Remove "1" dimensions from the strides list. */
function canonicalizeStrides(shape: number[], strides: number[]): number[] {
  const newStrides: number[] = [];
  for (let i = 0; i < shape.length; i++) {
    if (shape[i] === 1) newStrides.push(0);
    else newStrides.push(strides[i]);
  }
  return newStrides;
}

/** Get the strides for a shape in default row-major order. */
function defaultStrides(shape: number[]): number[] {
  if (shape.length === 0) return [];
  const strides = rep(shape.length, 1);
  for (let i = shape.length - 1; i > 0; i--) {
    strides[i - 1] = shape[i] * strides[i];
  }
  return strides;
}

/**
 * A multidimensional view into memory. An array can be thought of as the
 * combination of a linear buffer of memory, along with a `View`.
 */
class View {
  private constructor(
    /** The shape of the view (size of each dimension). */
    readonly shape: number[],

    /** How many indices to move in buffer for each hop in one dimension. */
    readonly strides: number[],

    /** Offset from the start of the buffer. */
    readonly offset: number,

    /** Masked out subarray where data is read. All other data is zeroed. */
    readonly mask: [number, number][] | null,
  ) {}

  static create(
    shape: number[],
    strides?: number[],
    offset: number = 0,
    mask: [number, number][] | null = null,
  ): View {
    if (shape.some((s) => s < 0))
      throw new Error("View shape must be non-negative");

    strides = strides
      ? canonicalizeStrides(shape, strides)
      : defaultStrides(shape);

    // Canonicalize zero-sized arrays.
    if (shape.includes(0)) {
      return new View(shape, rep(shape.length, 0), 0, null);
    }
    // Canonicalize default mask / no mask.
    if (mask !== null && mask.every(([b, e], i) => b === 0 && e === shape[i])) {
      mask = null;
    }
    // If dimension has size greater than 1, but is masked to only one index,
    // then set its stride to zero. Likewise, if any mask is empty, we can just
    // mask out the entire array.
    if (mask !== null) {
      const elimDims: number[] = [];
      let hasNoData = false;
      for (let i = 0; i < shape.length; i++) {
        const [b, e] = mask[i];
        if (b + 1 >= e) elimDims.push(i);
        if (b >= e) hasNoData = true;
      }
      if (elimDims.length) {
        if (hasNoData) {
          strides = rep(shape.length, 0);
          offset = 0;
          mask = rep(shape.length, () => [0, 0] as [number, number]);
        }
        for (const i of elimDims) {
          offset += strides[i] * mask[i][0];
          strides[i] = 0;
        }
      }
    }
    return new View(shape, strides, 0, null);
  }

  get size(): number {
    return this.shape.reduce((a, b) => a * b, 1);
  }

  /** Whether this is a default, contiguous, unaltered view of the data (identity). */
  get contiguous(): boolean {
    if (this.size === 0) {
      return true;
    }
    return (
      this.offset === 0 &&
      this.mask === null &&
      deepEqual(this.strides, defaultStrides(this.shape))
    );
  }

  /**
   * Try to compose this view with another one. `this` view is applied first,
   * followed by the argument. If this is not possible for the specific views,
   * return `null` instead.
   *
   * If composable, return a combined view with the same shape as `v1`.
   */
  compose(v1: View): View | null {
    const v2 = this;
    if (v2.contiguous) return v1;
    if (v1.contiguous) {
      if (deepEqual(v1.shape, v2.shape)) return v2;
      if (v1.size === v2.size) {
        const ret = v2.reshape(v1.shape);
        if (ret !== null) return ret;
      }
    }
    if (v1.mask !== null) {
      // Normalize out any masks in v1, applying them afterward.
      const newV1 = v1.shrink(v1.mask);
      const merged = v2.compose(newV1);
      return merged
        ? merged.pad(zip(v1.mask, v1.shape).map(([m, s]) => [m[0], s - m[1]]))
        : null;
    }

    // Project offset and strides.
    let origin = unravel(v2.shape, v1.offset); // v1 applies after v2
    let terms: [number, number][][] = rep(v2.shape.length, () => []);
    const strides = rep(v1.shape.length, 0);
    // TODO
  }

  /** Reshape the view into a new shape. */
  reshape(newShape: number[]): View | null {
    throw new Error("Not implemented");
  }
}

/**
 * Find position of `offset` in each dimension within an existing shape. Like
 * `numpy.unravel_index` in behavior.
 */
function unravel(shape: number[], offset: number): number[] {
  let acc = 1;
  const idxs: number[] = [];
  for (let i = shape.length - 1; i >= 0; i--) {
    const d = shape[i];
    idxs.push(Math.floor(offset / acc) % d);
    acc *= d;
  }
  return idxs.reverse();
}
