import { numpy as np } from "@jax-js/jax";
import { chain, scale } from "@jax-js/optax";
import { expect, test } from "vitest";

test("chain function combines transformations", () => {
  let params = np.array([1.0, 2.0, 3.0]);
  let updates = np.array([0.1, 0.2, 0.3]);

  // Chain two simple transformations
  const combined = chain(scale(2.0), scale(0.5));
  const state = combined.init(params.ref);

  const [newUpdates, _newState] = combined.update(
    updates.ref,
    state,
    params.ref,
  );

  // 2.0 * 0.5 = 1.0, so updates should be unchanged
  expect(newUpdates).toBeAllclose([0.1, 0.2, 0.3]);
});
