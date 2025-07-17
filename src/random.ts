// Port of the `jax.random` module.

import { bitcast, randomBits } from "./frontend/core";
import { array, Array, DType, scalar, stack } from "./numpy";

function validateKeyShape(key: Array): number[] {
  if (key.ndim === 0) {
    throw new Error("Key must have at least one dimension.");
  }
  if (key.shape[key.shape.length - 1] !== 2) {
    throw new Error(
      `Invalid key shape: ${key.shape}. Expected last dimension to be 2.`,
    );
  }
  return key.shape.slice(0, -1);
}

/** Create a pseudo-random number generator (PRNG) key from 32-bit integer seed. */
export function key(seed: number): Array {
  seed = seed >>> 0;
  // To match JAX, put the 32-bit seed into a 64-bit key in this way.
  return array([0, seed], { dtype: DType.Uint32 });
}

/** Splits a PRNG key into `num` new keys by adding a leading axis. */
export function split(key: Array, num: number | number[] = 2): Array {
  const shape = typeof num === "number" ? [num] : num;
  for (const len of shape) {
    if (len <= 0 || !Number.isInteger(len)) {
      throw new Error(
        `Invalid split length: ${len}. Must be a positive integer.`,
      );
    }
  }

  const keyShape = validateKeyShape(key);
  const k0 = key.ref.slice(...keyShape.map(() => null), 0);
  const k1 = key.slice(...keyShape.map(() => null), 1);
  return stack(
    // It's inefficient to calculate the PRNG key twice, then join the halves
    // together. But this allows us to avoid refactoring AluExp to support
    // multiple outputs, while remaining consistent with JAX.
    [
      randomBits(k0.ref, k1.ref, shape, 0) as Array,
      randomBits(k0, k1, shape, 1) as Array,
    ],
    -1,
  );
}

/** Sample uniform bits in the form of unsigned integers. */
export function bits(key: Array, shape: number[] = []): Array {
  const keyShape = validateKeyShape(key);
  return randomBits(
    key.ref.slice(...keyShape.map(() => null), 0),
    key.slice(...keyShape.map(() => null), 1),
    shape,
  ) as Array;
}

/** Sample uniform random values in [minval, maxval) with given shape. */
export function uniform(
  key: Array,
  shape: number[] = [],
  { minval = 0, maxval = 1 }: { minval?: number; maxval?: number } = {},
): Array {
  if (minval >= maxval) {
    throw new Error(`Invalid range: [${minval}, ${maxval}).`);
  }
  // Float32 has sign bit, 8 bits of exponent, and 23 bits of mantissa.
  const mantissa = bits(key, shape).div(
    scalar(1 << 9, { dtype: DType.Uint32, device: key.device }),
  );
  const float12 = mantissa.add(
    scalar(0x3f800000, { dtype: DType.Uint32, device: key.device }),
  ); // Add 1.0 in IEEE 754, now it's a float in [1, 2).
  const rand = bitcast(float12, DType.Float32).sub(1) as Array; // [0, 1) range
  if (minval === 0 && maxval === 1) {
    return rand;
  } else {
    return rand.mul(maxval - minval).add(minval);
  }
}
