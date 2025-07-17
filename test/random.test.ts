import { devices, init, numpy as np, random, setDevice } from "@jax-js/jax";
import { beforeEach, expect, suite, test } from "vitest";

const devicesAvailable = await init();

suite.each(devices)("device:%s", (device) => {
  const skipped = !devicesAvailable.includes(device);
  beforeEach(({ skip }) => {
    if (skipped) skip();
    setDevice(device);
  });

  test("random bits", () => {
    // jax.random.bits(jax.random.key(0))
    const x = random.bits(random.key(0));
    expect(x.shape).toEqual([]);
    expect(x.dtype).toEqual(np.uint32);
    expect(x.js()).toEqual(4070199207);

    // jax.random.bits(jax.random.key(0), shape=(4,))
    const y = random.bits(random.key(10), [4]);
    expect(y.shape).toEqual([4]);
    expect(y.dtype).toEqual(np.uint32);
    expect(y.js()).toEqual([169096361, 1572511259, 2689743692, 2228103506]);
  });

  test("random split is consistent with jax", () => {
    const splits = random.split(random.key(0), 3);
    expect(splits.shape).toEqual([3, 2]);
    expect(splits.dtype).toEqual(np.uint32);
    expect(splits.js()).toEqual([
      [1797259609, 2579123966],
      [928981903, 3453687069],
      [4146024105, 2718843009],
    ]);
  });

  test("generate uniform random", () => {
    const key = random.key(0);
    const [a, b, c] = random.split(key, 3);

    const x = random.uniform(a);
    expect(x.js()).toBeWithinRange(0, 1);

    const y = random.uniform(b, [0]);
    expect(y.js()).toEqual([]);

    const z = random.uniform(c, [2, 3]);
    expect(z.shape).toEqual([2, 3]);
    expect(z.dtype).toEqual(np.float32);
    const zx = z.js() as number[][];
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 3; j++) {
        expect(zx[i][j]).toBeWithinRange(0, 1);
      }
    }
  });
});
