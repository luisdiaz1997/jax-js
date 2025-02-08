import { expect, suite, test } from "vitest";
import { numpy as np, jvp, jacfwd, vmap } from "jax-js";

// test("has webgpu", async () => {
//   const adapter = await navigator.gpu?.requestAdapter();
//   const device = await adapter?.requestDevice();
//   if (!adapter || !device) {
//     throw new Error("No adapter or device");
//   }
//   console.log(device.adapterInfo.architecture);
//   console.log(device.adapterInfo.vendor);
//   console.log(adapter.limits.maxVertexBufferArrayStride);
// });

/** Take the derivative of a simple function. */
function deriv(f: (x: np.Array) => np.Array): (x: np.ArrayLike) => np.Array {
  return (x) => {
    const [_y, dy] = jvp(f, [x], [1.0]);
    return dy;
  };
}

test("can create array", () => {
  const x = np.array([1, 2, 3]);
  expect(x.js()).toEqual([1, 2, 3]);
});

suite("jax.jvp()", () => {
  test("can take scalar derivatives", () => {
    const x = 3.0;
    expect(np.sin(x)).toBeAllclose(0.141120001);
    expect(deriv(np.sin)(x)).toBeAllclose(-0.989992499);
    expect(deriv(deriv(np.sin))(x)).toBeAllclose(-0.141120001);
    expect(deriv(deriv(deriv(np.sin)))(x)).toBeAllclose(0.989992499);
  });

  test("can take jvp of pytrees", () => {
    const result = jvp(
      (x: { a: np.Array; b: np.Array }) => x.a.mul(x.a).add(x.b),
      [{ a: 1, b: 2 }],
      [{ a: 1, b: 0 }]
    );
    expect(result[0]).toBeAllclose(3);
    expect(result[1]).toBeAllclose(2);
  });

  test("works for vector to scalar functions", () => {
    const f = (x: np.Array) => np.reduceSum(x);
    const x = np.array([1, 2, 3]);
    expect(f(x)).toBeAllclose(6);
    expect(jvp(f, [x], [np.array([1, 1, 1])])[1]).toBeAllclose(3);
  });
});

// suite("jax.vmap()", () => {});

suite("jax.jacfwd()", () => {
  test("computes jacobian of 3d square", () => {
    const f = (x: np.Array) => x.mul(x);
    const x = np.array([1, 2, 3]);
    const j = jacfwd(f, x);
    expect(j).toBeAllclose(
      np.array([
        [2, 0, 0],
        [0, 4, 0],
        [0, 0, 6],
      ])
    );
  });
});
