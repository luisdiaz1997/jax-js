import { beforeEach, expect, suite, test } from "vitest";

import { backendTypes, init, setBackend } from "../backend";
import { Array, array } from "./array";

const backendsAvailable = await init("cpu");

suite.each(backendTypes)("backend:%s", (backend) => {
  const skipped = !backendsAvailable.includes(backend);

  beforeEach(({ skip }) => {
    if (skipped) skip();
    setBackend(backend);
  });

  test("can construct Array.zeros()", async () => {
    const ar = Array.zeros([3, 3]);
    expect(ar.shape).toEqual([3, 3]);
    expect(ar.dtype).toEqual("float32");
    expect(await ar.data()).toEqual(
      new Float32Array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
    );
    expect(await ar.transpose().data()).toEqual(
      new Float32Array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
    );
    expect(ar.transpose().dataSync()).toEqual(
      new Float32Array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
    );
  });

  test("can construct Array.ones()", async () => {
    const ar = Array.ones([2, 2]);
    expect(ar.shape).toEqual([2, 2]);
    expect(ar.dtype).toEqual("float32");
    expect(await ar.data()).toEqual(new Float32Array([1, 1, 1, 1]));
  });

  test("can add two arrays", async () => {
    const ar1 = Array.ones([2, 2]);
    const ar2 = Array.ones([2, 2]);
    const ar3 = ar1.add(ar2);
    expect(ar3.shape).toEqual([2, 2]);
    expect(ar3.dtype).toEqual("float32");
    expect(await ar3.data()).toEqual(new Float32Array([2, 2, 2, 2]));
  });

  test("can construct arrays from data", () => {
    const a = array([1, 2, 3, 4]);
    const b = array([10, 5, 2, -8.5]);
    const c = a.mul(b);
    expect(c.shape).toEqual([4]);
    expect(c.dtype).toEqual("float32");
    expect(c.dataSync()).toEqual(new Float32Array([10, 10, 6, -34]));
    expect(c.reshape([2, 2]).transpose().dataSync()).toEqual(
      new Float32Array([10, 6, 10, -34]),
    );
  });

  test("can add array to itself", () => {
    const a = array([1, 2, 3]);
    // Make sure duplicate references don't trip up the backend.
    const b = a.add(a).add(a);
    expect(b.dataSync()).toEqual(new Float32Array([3, 6, 9]));
  });

  test("can coerce array to primitive", () => {
    const a = array(42);
    expect(a).toBeCloseTo(42);

    // https://github.com/microsoft/TypeScript/issues/42218
    expect(+(a as any)).toEqual(42);
    expect((a as any) + 1).toEqual(43);
    expect((a as any) ** 2).toEqual(42 ** 2);
  });

  test("construct bool array", () => {
    const a = array([true, false, true]);
    expect(a.shape).toEqual([3]);
    expect(a.dtype).toEqual("bool");

    expect(a.dataSync()).toEqual(new Int32Array([1, 0, 1]));
    expect(a.js()).toEqual([true, false, true]);

    const b = array([1, 3, 4]);
    expect(b.gt(2).js()).toEqual([false, true, true]);
  });
});
