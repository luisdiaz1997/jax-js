import { grad, numpy as np } from "@jax-js/jax";
// import { applyUpdates, l2Loss, sgd } from "@jax-js/optax";
import { expect, test } from "vitest";

/*
test("stochastic gradient descent", () => {
  const solver = sgd({ learningRate: 0.1 });
  let params = np.array([1.0, 2.0, 3.0]);
  let optState = solver.init(params);
  let updates: np.Array;

  const f = (x: np.Array) => l2Loss(x, np.ones([3]));
  const paramsGrad = grad(f)(params);
  [updates, optState] = solver.update(paramsGrad, optState, params);
  params = applyUpdates(params, updates);

  expect(params).toBeAllclose([0.8, 1.6, 2.4]);
});
*/

test("placeholder test", () => {
  const f = grad((x: np.Array) => x.sum());
  expect(f(np.array([1, 2, 3])).js()).toEqual([1, 1, 1]);
});
