import { describe, expect, test as globalTest } from "vitest";
import { backendTypes, init } from "./backend";
import { Array } from "./array";

const backendsAvailable = await init(...backendTypes);

describe.each(backendTypes)("Backend '%s'", (backend) => {
  const skipped = !backendsAvailable.includes(backend);
  const test = globalTest.skipIf(skipped);

  test("can construct Array.zeros()", async () => {
    const ar = Array.zeros([3, 3], { backend });
    expect(ar.shape).toEqual([3, 3]);
    expect(ar.dtype).toEqual("float32");
    expect(await ar.data()).toEqual(
      new Float32Array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
    );
  });

  test("can construct Array.ones()", async () => {
    const ar = Array.ones([2, 2], { backend });
    expect(ar.shape).toEqual([2, 2]);
    expect(ar.dtype).toEqual("float32");
    expect(await ar.data()).toEqual(new Float32Array([1, 1, 1, 1]));
  });

  test("can add two arrays", async () => {
    const ar1 = Array.ones([2, 2], { backend });
    const ar2 = Array.ones([2, 2], { backend });
    const ar3 = ar1.add(ar2);
    expect(ar3.shape).toEqual([2, 2]);
    expect(ar3.dtype).toEqual("float32");
    expect(await ar3.data()).toEqual(new Float32Array([2, 2, 2, 2]));
  });
});
