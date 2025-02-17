import * as tf from "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-core/dist/public/chained_ops/register_all_chained_ops";
import "@tensorflow/tfjs-core/dist/register_all_gradients";
import "@tensorflow/tfjs-backend-cpu";
import { deepEqual, range, unzip2, zip } from "./utils";
import {
  JsTreeDef,
  flatten as treeFlatten,
  unflatten as treeUnflatten,
} from "./tree";

export enum DType {
  Float32 = "float32",
  Int32 = "int32",
  Bool = "bool",
  Complex64 = "complex64",
}

export enum Primitive {
  Add = "add",
  Mul = "mul",
  Neg = "neg",
  Sin = "sin",
  Cos = "cos",
  ReduceSum = "reduce_sum",
  Greater = "greater",
  Less = "less",
  Transpose = "transpose",
  Broadcast = "broadcast",
}

export function add(x: TracerValue, y: TracerValue) {
  return bind1(Primitive.Add, [x, y]);
}

export function mul(x: TracerValue, y: TracerValue) {
  return bind1(Primitive.Mul, [x, y]);
}

export function neg(x: TracerValue) {
  return bind1(Primitive.Neg, [x]);
}

export function sin(x: TracerValue) {
  return bind1(Primitive.Sin, [x]);
}

export function cos(x: TracerValue) {
  return bind1(Primitive.Cos, [x]);
}

export function greater(x: TracerValue, y: TracerValue) {
  return bind1(Primitive.Greater, [x, y]);
}

export function less(x: TracerValue, y: TracerValue) {
  return bind1(Primitive.Less, [x, y]);
}

export function transpose(x: TracerValue, perm?: number[]) {
  return bind1(Primitive.Transpose, [x], { perm });
}

export function broadcast(x: TracerValue, shape: number[], axes: number[]) {
  return bind1(Primitive.Broadcast, [x], { shape, axes });
}

export function reduceSum(x: TracerValue, axis?: number | number[]) {
  if (axis === null) {
    if (x instanceof Tracer) {
      axis = [...JsArray(x.shape.length).keys()];
    } else {
      axis = [];
    }
  }
  if (typeof axis === "number") {
    axis = [axis];
  }
  return bind1(Primitive.ReduceSum, [x], { axis });
}

function bind1(
  prim: Primitive,
  args: TracerValue[],
  params: Record<string, any> = {}
) {
  const [results] = bind(prim, args, params);
  return results;
}

type MainTrace = {
  level: number;
  traceType: new (main: MainTrace) => Trace; // Concrete Trace subclass.
  globalData: any | null;
};

let traceStack: MainTrace[] = [];
let dynamicTrace: MainTrace | null = null;

// Push an interpreter onto the trace stack. Use this like:
// `using main = newMain(...);`
function newMain(
  traceType: any,
  globalData: any | null = null
): Disposable & MainTrace {
  const level = traceStack.length;
  const main = { level, traceType, globalData };
  traceStack.push(main);
  return Object.assign(main, {
    [Symbol.dispose]() {
      traceStack.pop();
    },
  });
}

type TracerValue = Tracer | number | boolean;

abstract class Trace {
  constructor(public main: MainTrace) {}

  abstract pure(val: TracerValue): Tracer;
  abstract lift(val: Tracer): Tracer;

  abstract processPrimitive(
    primitive: Primitive,
    tracers: Tracer[],
    params: Record<string, any>
  ): Tracer[];
}

interface AbstractValue {
  shape: number[];
  dtype: DType;

  _neg: (x: Tracer) => Tracer;
  _add: (x: Tracer, y: Tracer) => Tracer;
  _mul: (x: Tracer, y: Tracer) => Tracer;
  _gt: (x: Tracer, y: Tracer) => Tracer;
  _lt: (x: Tracer, y: Tracer) => Tracer;
}

abstract class Tracer {
  readonly _trace: Trace;

  constructor(trace: Trace) {
    this._trace = trace;
  }

  abstract get aval(): AbstractValue;
  abstract toString(): string;

  get shape() {
    return this.aval.shape;
  }

  fullLower(): Tracer {
    return this; // default implementation
  }

  // These types aren't technically correct since they don't account for the
  // fact that tracers can be lifted to different levels. But they simplify the
  // API visible to users.
  neg() {
    return this.aval._neg(this) as this;
  }
  add(other: this | TracerValue) {
    return this.aval._add(this, pureArray(other)) as this;
  }
  mul(other: this | TracerValue) {
    return this.aval._mul(this, pureArray(other)) as this;
  }
  gt(other: this | TracerValue) {
    return this.aval._gt(this, pureArray(other)) as this;
  }
  lt(other: this | TracerValue) {
    return this.aval._lt(this, pureArray(other)) as this;
  }
}

export function ndim(x: TracerValue) {
  if (x instanceof Tracer) {
    return x.shape.length;
  } else {
    return 0;
  }
}

const JsArray = globalThis.Array;

class ShapedArray implements AbstractValue {
  readonly arrayAbstractionLevel: number = 1;

  constructor(
    public readonly shape: number[],
    public readonly dtype: DType
  ) {}

  static fromAval(aval: AbstractValue) {
    return new ShapedArray(aval.shape, aval.dtype);
  }

  get ndim() {
    return this.shape.length;
  }

  // See note about primitive wrappers with fudged types.
  _neg = neg as any;
  _add = add as any;
  _mul = mul as any;
  _gt = greater as any;
  _lt = less as any;

  strShort() {
    return `${this.dtype}[${this.shape.join(",")}]`;
  }

  equals(other: ShapedArray) {
    return (
      this === other ||
      (this.constructor === other.constructor &&
        this.ndim === other.ndim &&
        this.shape.every((d, i) => d === other.shape[i]))
    );
  }
}

class ConcreteArray extends ShapedArray {
  readonly arrayAbstractionLevel: number = 2;

  constructor(public readonly val: tf.Tensor) {
    super(val.shape, val.dtype as any);
  }
}

/**
 * Equivalent to `jnp.Array` from JAX, a tensor type.
 *
 * Not to be confused with the JavaScript "Array" constructor. Avoid importing this into your code's
 * namespace if you're already using the JavaScript "Array" type by name.
 */
export class Array extends Tracer {
  readonly dtype: DType;

  constructor(public readonly data: tf.Tensor) {
    super(baseArrayTrace);
    if (Object.values(DType).includes(data.dtype as any)) {
      this.dtype = data.dtype as DType;
    } else {
      throw new TypeError(`Unsupported dtype: ${data.dtype}`);
    }
  }

  get aval(): AbstractValue {
    return new ConcreteArray(this.data);
  }

  /** Return a simple string representation of the array's dimensions. */
  toString(): string {
    return `Array[${this.data.shape.join(", ")}]`;
  }

  /** Convert this array into a JavaScript object (blocking). */
  js() {
    return this.data.arraySync();
  }

  /** Convert this array into a JavaScript object, asynchronously. */
  async jsAsync() {
    return await this.data.array();
  }
}

/** If x is a value, lift it into an array, otherwise leave it be. */
function pureArray(x: TracerValue): Tracer {
  if (x instanceof Tracer) {
    return x;
  } else {
    return new Array(tf.scalar(x));
  }
}

function getAval(x: TracerValue): AbstractValue {
  if (x instanceof Tracer) {
    return x.aval;
  } else if (typeof x === "boolean" || typeof x === "number") {
    return new ConcreteArray(tf.scalar(x));
  } else {
    throw new TypeError(`Unknown value: ${x}`);
  }
}

function bind(
  prim: Primitive,
  args: TracerValue[],
  params: Record<string, any> = {}
) {
  const topTrace = findTopTrace(args);
  const tracers = args.map((arg) => fullRaise(topTrace, arg));
  const outs = topTrace.processPrimitive(prim, tracers, params);
  // console.info(`processing rule for ${prim} on ${tracers} and got ${outs}`);
  return outs.map((out) => out.fullLower());
}

function findTopTrace(xs: TracerValue[]): Trace {
  let topMain: MainTrace = traceStack[0];
  for (const x of xs) {
    if (x instanceof Tracer && x._trace.main.level > topMain.level) {
      topMain = x._trace.main;
    }
  }
  if (dynamicTrace && dynamicTrace.level > topMain.level) {
    topMain = dynamicTrace;
  }
  return new topMain.traceType(topMain);
}

function fullRaise(trace: Trace, val: TracerValue): Tracer {
  if (!(val instanceof Tracer)) {
    // remember to assert type(val) in jax_types
    return trace.pure(val);
  }
  const level = trace.main.level;
  if (Object.is(val._trace.main, trace.main)) {
    return val;
  } else if (val._trace.main.level < level) {
    return trace.lift(val);
  } else if (val._trace.main.level > level) {
    throw new Error(
      `Can't lift Tracer level ${val._trace.main.level} to level ${level}`
    );
  } else {
    throw new Error(`Different traces at same level: ${val._trace}, ${trace}.`);
  }
}

class EvalTrace extends Trace {
  // No boxing in Tracers needed.
  pure = (x: TracerValue) => pureArray(x);
  lift = (x: Tracer) => x;

  processPrimitive(
    primitive: Primitive,
    tracers: Tracer[],
    params: Record<string, any>
  ): Tracer[] {
    return implRules[primitive](tracers as Array[], params);
  }
}

// Special bottom of the stack.
traceStack.push({ level: 0, traceType: EvalTrace, globalData: null });
const baseArrayTrace = new EvalTrace(traceStack[0]);

type ImplRule = (tracers: Array[], params: any) => Array[];

const implRules: Record<Primitive, ImplRule> = {
  [Primitive.Add]([x, y]) {
    return [new Array(tf.add(x.data, y.data))];
  },
  [Primitive.Mul]([x, y]) {
    return [new Array(tf.mul(x.data, y.data))];
  },
  [Primitive.Neg]([x]) {
    return [new Array(tf.neg(x.data))];
  },
  [Primitive.Sin]([x]) {
    return [new Array(tf.sin(x.data))];
  },
  [Primitive.Cos]([x]) {
    return [new Array(tf.cos(x.data))];
  },
  [Primitive.ReduceSum]([x], { axis }: { axis: number[] }) {
    return [new Array(tf.sum(x.data, axis))];
  },
  [Primitive.Greater]([x, y]) {
    return [new Array(tf.greater(x.data, y.data))];
  },
  [Primitive.Less]([x, y]) {
    return [new Array(tf.less(x.data, y.data))];
  },
  [Primitive.Transpose]([x], { perm }: { perm?: number[] }) {
    return [new Array(tf.transpose(x.data, perm))];
  },
  [Primitive.Broadcast](
    [x],
    { shape, axes }: { shape: number[]; axes: number[] }
  ) {
    let data = x.data;
    for (const axis of axes.toSorted()) {
      data = tf.expandDims(data, axis);
    }
    return [new Array(tf.broadcastTo(data, shape))];
  },
};

function zerosLike(val: TracerValue): Array {
  const aval = getAval(val);
  return new Array(tf.zeros(aval.shape, aval.dtype));
}

class JVPTracer extends Tracer {
  constructor(
    trace: Trace,
    public readonly primal: Tracer,
    public readonly tangent: Tracer
  ) {
    super(trace);
  }

  get aval(): AbstractValue {
    return this.primal.aval;
  }

  toString(): string {
    return `JVPTracer(${this.primal}, ${this.tangent})`;
  }
}

class JVPTrace extends Trace {
  pure(val: TracerValue) {
    return this.lift(pureArray(val));
  }

  lift(val: Tracer): Tracer {
    return new JVPTracer(this, val, zerosLike(val));
  }

  processPrimitive(
    primitive: Primitive,
    tracers: JVPTracer[],
    params: Record<string, any>
  ): JVPTracer[] {
    const [primalsIn, tangentsIn] = unzip2(
      tracers.map((x) => [x.primal, x.tangent])
    );
    const jvpRule: JvpRule | undefined = jvpRules[primitive];
    if (jvpRule === undefined) {
      throw new Error(`No JVP rule for: ${primitive}`);
    }
    const [primalsOut, tangentsOut] = jvpRule(primalsIn, tangentsIn, params);
    return zip(primalsOut, tangentsOut).map(
      ([x, t]) => new JVPTracer(this, x, t)
    );
  }
}

type JvpRule = (
  primal: Tracer[],
  tangents: Tracer[],
  params: any
) => [Tracer[], Tracer[]];

const jvpRules: Partial<Record<Primitive, JvpRule>> = {
  [Primitive.Add]([x, y], [dx, dy]) {
    return [[x.add(y)], [dx.add(dy)]];
  },
  [Primitive.Mul]([x, y], [dx, dy]) {
    return [[x.mul(y)], [x.mul(dy).add(dx.mul(y))]];
  },
  [Primitive.Neg]([x], [dx]) {
    return [[x.neg()], [dx.neg()]];
  },
  [Primitive.Sin]([x], [dx]) {
    return [[sin(x)], [cos(x).mul(dx)]];
  },
  [Primitive.Cos]([x], [dx]) {
    return [[cos(x)], [neg(sin(x)).mul(dx)]];
  },
  [Primitive.ReduceSum]([x], [dx], { axis }: { axis: number[] }) {
    return [[reduceSum(x, axis)], [reduceSum(dx, axis)]];
  },
  [Primitive.Greater]([x, y], _tangents) {
    const outPrimal = greater(x, y);
    return [[outPrimal], [zerosLike(outPrimal)]];
  },
  [Primitive.Less]([x, y], _tangents) {
    const outPrimal = less(x, y);
    return [[outPrimal], [zerosLike(outPrimal)]];
  },
};

function jvpFlat(
  f: (...x: TracerValue[]) => TracerValue[],
  primals: TracerValue[],
  tangents: TracerValue[]
): [Tracer[], Tracer[]] {
  using main = newMain(JVPTrace);
  // console.info("creating new jvp main", traceStack);
  const trace = new JVPTrace(main);
  const tracersIn = zip(primals, tangents).map(
    ([x, t]) => new JVPTracer(trace, pureArray(x), pureArray(t))
  );
  const outs = f(...tracersIn);
  const tracersOut = outs.map((out) => fullRaise(trace, out) as JVPTracer);
  return unzip2(tracersOut.map((t) => [t.primal, t.tangent]));
}

export function jvp(
  f: (...x: any[]) => any,
  primals: any[],
  tangents: any[]
): [any, any] {
  const [primalsFlat, inTree] = treeFlatten(primals);
  const [tangentsFlat, inTree2] = treeFlatten(tangents);
  if (!inTree.equals(inTree2)) {
    throw new TypeError("Mismatched tree structures in jvp");
  }

  const [flatFun, outTree] = flattenFun(f, inTree);

  const [primalsOutFlat, tangentsOutFlat] = jvpFlat(
    flatFun,
    primalsFlat,
    tangentsFlat
  );
  if (outTree.value === undefined) {
    throw new Error("outTree was not set in jvp");
  }
  const primalsOut = treeUnflatten(outTree.value, primalsOutFlat);
  const tangentsOut = treeUnflatten(outTree.value, tangentsOutFlat);
  return [primalsOut, tangentsOut];
}

function flattenFun(
  f: any,
  inTree: JsTreeDef
): [any, { value: JsTreeDef | undefined }] {
  const store: { value: JsTreeDef | undefined } = { value: undefined };
  const flatFun = (...argsFlat: any[]) => {
    const pytreeArgs = treeUnflatten(inTree, argsFlat);
    const out = f(...pytreeArgs);
    const [outFlat, outTree] = treeFlatten(out);
    store.value = outTree;
    return outFlat;
  };
  return [flatFun, store];
}

// vmap() implementation begins

function mappedAval(batchDim: number, aval: AbstractValue) {
  const shape = [...aval.shape];
  shape.splice(batchDim, 1);
  return new ShapedArray(shape, aval.dtype);
}

function moveBatchAxis(
  axisSize: number,
  src: number | null,
  dst: number,
  x: Tracer
) {
  if (src === null) {
    // not_mapped
    const targetShape = [...x.shape];
    targetShape.splice(dst, 0, axisSize);
    return broadcast(x, targetShape, [dst]);
  } else if (src === dst) {
    return x;
  } else {
    return moveaxis(x, src, dst);
  }
}

/** Move one axis to a different index. */
export function moveaxis(x: TracerValue, src: number, dst: number) {
  const t = pureArray(x);
  const perm = [...JsArray(t.shape.length).keys()];
  perm.splice(src, 1);
  perm.splice(dst, 0, src);
  return transpose(t, perm);
}

class BatchTracer extends Tracer {
  constructor(
    trace: Trace,
    public readonly val: Tracer,
    public readonly batchDim: number | null
  ) {
    super(trace);
  }

  get aval(): AbstractValue {
    if (this.batchDim === null) {
      return this.val.aval;
    } else {
      return mappedAval(this.batchDim, this.val.aval);
    }
  }

  toString(): string {
    return `BatchTracer(${this.val}, ${this.batchDim})`;
  }

  fullLower(): Tracer {
    if (this.batchDim === null) {
      return this.val.fullLower();
    } else {
      return this;
    }
  }
}

class BatchTrace extends Trace {
  pure(val: TracerValue) {
    return this.lift(pureArray(val));
  }

  lift(val: Tracer): Tracer {
    return new BatchTracer(this, val, null);
  }

  processPrimitive(
    primitive: Primitive,
    tracers: BatchTracer[],
    params: Record<string, any>
  ): BatchTracer[] {
    const [valsIn, bdimsIn] = unzip2(tracers.map((t) => [t.val, t.batchDim]));
    const vmapRule = vmapRules[primitive];
    if (vmapRule === undefined) {
      throw new Error(`No vmap rule for: ${primitive}`);
    }
    const [valOuts, bdimOuts] = vmapRule(
      this.axisSize,
      valsIn,
      bdimsIn,
      params
    );
    return zip(valOuts, bdimOuts).map(
      ([x, bd]) => new BatchTracer(this, x, bd)
    );
  }

  get axisSize(): number {
    return this.main.globalData;
  }
}

type VmapRule = (
  axisSize: number,
  valsIn: Tracer[],
  dimsIn: (number | null)[],
  params: any
) => [Tracer[], (number | null)[]];

function handleScalarBroadcasting(nd: number, x: Tracer, d: number | null) {
  if (d === null || nd === ndim(x)) {
    return x;
  } else {
    const axes = range(ndim(x), nd);
    const shape = [...x.shape, ...axes.map(() => 1)];
    return broadcast(x, shape, axes);
  }
}

/** Process a primitive with built-in broadcasting. */
function broadcastBatcher(op: (...x: Tracer[]) => Tracer) {
  return (
    axisSize: number,
    args: Tracer[],
    dims: (number | null)[]
  ): ReturnType<VmapRule> => {
    if (args.length === 0) {
      throw new Error("Empty list in broadcastBatcher");
    }

    const idx = dims.findIndex((d) => d !== null);
    if (idx === -1) {
      // No-op case: no mapped indices, just pass it down to the parent tracer.
      return [[op(...args)], [null]];
    }
    if (
      // If only agreeing batch dims, as well as scalars, just call the primitive.
      zip(args, dims).every(
        ([x, d]) =>
          ndim(x) === 0 ||
          (deepEqual(x.shape, args[idx].shape) && d === dims[idx])
      )
    ) {
      return [[op(...args)], [dims[idx]]];
    }

    args = args.map((x, i) =>
      ndim(x) > 0 ? moveBatchAxis(axisSize, dims[i], 0, x) : x
    );
    // Now the batch axis has been added to the front. Handle special-case of
    // scalar broadcasting, since unmapped axes may have a singleton axis
    // inserted and then rely on the built-in broadcasting of the primitive.
    const nd = Math.max(...args.map(ndim));
    args = args.map((x, i) => handleScalarBroadcasting(nd, x, dims[i]));
    return [[op(...args)], [0]];
  };
}

function vectorizedUnopBatchingRule(op: (x: Tracer) => Tracer) {
  return (
    axisSize: number,
    [x]: Tracer[],
    [xBdim]: (number | null)[]
  ): ReturnType<VmapRule> => {
    return [[op(x)], [xBdim]];
  };
}

const vmapRules: Partial<Record<Primitive, VmapRule>> = {
  [Primitive.Add]: broadcastBatcher(add),
  [Primitive.Mul]: broadcastBatcher(mul),
  [Primitive.Neg]: vectorizedUnopBatchingRule(neg),
  [Primitive.Sin]: vectorizedUnopBatchingRule(sin),
  [Primitive.Cos]: vectorizedUnopBatchingRule(cos),
  [Primitive.ReduceSum](
    axisSize: number,
    [x]: Tracer[],
    [xBdim]: (number | null)[],
    { axis }: { axis: number[] }
  ): ReturnType<VmapRule> {
    if (xBdim === null) {
      return [[reduceSum(x, axis)], [null]];
    }
    const newAxis = axis.map((ax) => ax + (xBdim <= ax ? 1 : 0));
    const outBdim = xBdim - axis.filter((ax) => ax < xBdim).length;
    return [[reduceSum(x, newAxis)], [outBdim]];
  },
};

function vmapFlat(
  f: (...x: TracerValue[]) => TracerValue[],
  inAxes: number[],
  args: TracerValue[]
): Tracer[] {
  let axisSize: number | undefined = undefined;
  for (let i = 0; i < args.length; i++) {
    if (inAxes[i] !== null) {
      const arg = args[i];
      if (!(arg instanceof Tracer)) {
        throw new TypeError("vmap requires Tracer argument for mapped axes");
      }
      const size = arg.shape[inAxes[i]];
      if (axisSize === undefined) {
        axisSize = size;
      } else if (axisSize !== size) {
        throw new TypeError(
          "vmap requires all mapped axes to have the same size"
        );
      }
    }
  }
  if (axisSize === undefined) {
    throw new TypeError("vmap requires at least one mapped axis");
  }

  let valsOut: Tracer[], bdimsOut: (number | null)[];
  {
    using main = newMain(BatchTrace, axisSize);
    // console.info("creating new vmap main", traceStack);
    const trace = new BatchTrace(main);
    const tracersIn = args.map((x, i) =>
      inAxes[i] === null
        ? pureArray(x)
        : new BatchTracer(trace, pureArray(x), inAxes[i])
    );
    const outs = f(...tracersIn);
    const tracersOut = outs.map((out) => fullRaise(trace, out) as BatchTracer);
    [valsOut, bdimsOut] = unzip2(tracersOut.map((t) => [t.val, t.batchDim]));
  }
  return zip(valsOut, bdimsOut).map(([valOut, bdim]) =>
    moveBatchAxis(axisSize, bdim, 0, valOut)
  ); // outs_transposed
}

export function vmap(
  f: (...x: any[]) => any,
  inAxes: any[]
): (...x: any[]) => any {
  return (...args: any[]) => {
    const [argsFlat, inTree] = treeFlatten(args);
    const [inAxesFlat, inTree2] = treeFlatten(inAxes);
    if (!inTree.equals(inTree2)) {
      throw new TypeError("Mismatched tree structures in vmap");
    }
    const [fFlat, outTree] = flattenFun(f, inTree);
    const outsFlat = vmapFlat(fFlat, inAxesFlat, argsFlat);
    if (outTree.value === undefined) {
      throw new Error("outTree was not set in vmap");
    }
    return treeUnflatten(outTree.value, outsFlat);
  };
}

export function jacfwd(f: any, x: Tracer) {
  if (x.shape.length !== 1) {
    throw new TypeError("jacfwd only supports 1D inputs");
  }
  const [size] = x.shape;
  const pushfwd = (v: Tracer) => jvp(f, [x], [v])[1];
  return vmap(pushfwd, [0])(new Array(tf.eye(size)));
}

/** Variable in a Jaxpr expression. */
export class Var {
  static nextId = 1; // For debugging, since JavaScript has no id() function like Python.

  readonly id: number;
  readonly aval: ShapedArray;

  constructor(aval: ShapedArray) {
    this.id = Var.nextId++;
    this.aval = aval;
  }
}

/** Literal in a Jaxpr expression. */
export class Lit {
  readonly val: Array;
  readonly aval: ShapedArray;

  constructor(val: Array | number | boolean) {
    this.aval = ShapedArray.fromAval(getAval(val));
    const ar = pureArray(val);
    if (!(ar instanceof Array)) {
      throw new TypeError("Lit only supports defined Array values");
    }
    this.val = ar;
  }
}

export type Atom = Var | Lit;

export type JaxprEqn = {
  primitive: Primitive;
  inputs: Atom[];
  params: Record<string, any>;
  outBinders: Var[];
};

export type Jaxpr = {
  inBinders: Var[];
  eqns: JaxprEqn[];
  outs: Atom[];
};

export class JaxprType {
  constructor(
    readonly inTypes: ShapedArray[],
    readonly outTypes: ShapedArray[]
  ) {}

  toString(): string {
    const inTypes = this.inTypes.map((aval) => aval.strShort()).join(", ");
    const outTypes = this.outTypes.map((aval) => aval.strShort()).join(", ");
    return `(${inTypes}) -> (${outTypes})`;
  }
}

function typecheckJaxpr(jaxpr: Jaxpr): JaxprType {
  const env = new Set<Var>();

  for (const v of jaxpr.inBinders) {
    if (env.has(v)) {
      throw new Error(`Duplicate variable binding: ${v}`);
    }
    env.add(v);
  }

  for (const eqn of jaxpr.eqns) {
    const inTypes = eqn.inputs.map((x) => typecheckAtom(env, x));
    const outTypes = abstractEvalRules[eqn.primitive](inTypes, eqn.params);
    for (const [outBinder, outType] of zip(eqn.outBinders, outTypes)) {
      if (!outType.equals(outBinder.aval)) {
        throw new TypeError(
          `Output binder type mismatch in ${eqn.primitive}: ${outBinder} vs ${outType}`
        );
      }
      if (env.has(outBinder)) {
        throw new Error(`Duplicate variable binding: ${outBinder}`);
      }
      env.add(outBinder);
    }
  }

  const inTypes = jaxpr.inBinders.map((v) => v.aval);
  const outTypes = jaxpr.outs.map((x) => typecheckAtom(env, x));
  return new JaxprType(inTypes, outTypes);
}

function typecheckAtom(env: Set<Var>, x: Atom): ShapedArray {
  if (x instanceof Var) {
    if (!env.has(x)) {
      throw new Error(`Unknown variable: ${x}`);
    }
    return x.aval;
  } else if (x instanceof Lit) {
    return x.aval;
  } else {
    throw new TypeError(`Invalid atom type: ${x}`);
  }
}

/** Evaluate a jaxpr on an array of inputs. */
function evalJaxpr(jaxpr: Jaxpr, args: Tracer[]): Tracer[] {
  const env = new Map<Var, Tracer>();

  const read = (x: Atom) => (x instanceof Var ? env.get(x)! : x.val);
  const write = (v: Var, val: Tracer) => {
    if (env.has(v)) throw new Error(`Variable already bound: ${v}`);
    env.set(v, val);
  };

  for (const [v, arg] of zip(jaxpr.inBinders, args)) write(v, arg);
  for (const eqn of jaxpr.eqns) {
    const inVals = eqn.inputs.map(read);
    const outVals = bind(eqn.primitive, inVals, eqn.params);
    for (const [v, val] of zip(eqn.outBinders, outVals)) write(v, val);
  }
  return jaxpr.outs.map(read);
}

function jaxprAsFun(jaxpr: Jaxpr) {
  return (...args: Tracer[]) => evalJaxpr(jaxpr, args);
}

class JaxprTracer extends Tracer {
  constructor(
    trace: Trace,
    readonly aval: ShapedArray
  ) {
    super(trace);
  }

  toString(): string {
    return `JaxprTracer(${this.aval.strShort()})`;
  }
}

// TODO: JaxprTrace

type AbstractEvalRule = (shapes: ShapedArray[], params: any) => ShapedArray[];

/**
 * Implements a NumPy-style generalized broadcast rule on two array shapes.
 *
 * "When operating on two arrays, NumPy compares their shapes element-wise. It starts with the
 * trailing (i.e. rightmost) dimension and works its way left. Two dimensions are compatible when:
 *   1. they are equal, or
 *   2. one of them is 1."
 *
 * Throws a TypeError if the broadcast is not possible.
 *
 * <https://numpy.org/doc/stable/user/basics.broadcasting.html#general-broadcasting-rules>
 */
function generalBroadcast(a: number[], b: number[]): number[] {
  const out: number[] = [];
  let i = a.length - 1;
  let j = b.length - 1;
  for (; i >= 0 && j >= 0; i--, j--) {
    const x = a[i];
    const y = b[j];
    if (x === y) {
      out.push(x);
    } else if (x === 1) {
      out.push(y);
    } else if (y === 1) {
      out.push(x);
    } else {
      throw new TypeError(`Incompatible array broadcast shapes: ${a} vs ${b}`);
    }
  }
  for (; i >= 0; i--) {
    out.push(a[i]);
  }
  for (; j >= 0; j--) {
    out.push(b[j]);
  }
  return out.reverse();
}

function binopAbstractEval([x, y]: ShapedArray[]) {
  if (!(x instanceof ShapedArray) || !(y instanceof ShapedArray)) {
    throw new TypeError("binopAbstractEval expects ShapedArray inputs");
  }
  if (x.dtype !== y.dtype) {
    // TODO: Relax this restriction on dtype equality, or add automatic casts.
    throw new TypeError(`Mismatched dtypes: ${x.dtype} vs ${y.dtype}`);
  }
  return [new ShapedArray(generalBroadcast(x.shape, y.shape), x.dtype)];
}

function compareAbstractEval([x, y]: ShapedArray[]) {
  if (!(x instanceof ShapedArray) || !(y instanceof ShapedArray)) {
    throw new TypeError("binopAbstractEval expects ShapedArray inputs");
  }
  if (x.dtype !== y.dtype) {
    // TODO: Relax this restriction on dtype equality, or add automatic casts.
    throw new TypeError(`Mismatched dtypes: ${x.dtype} vs ${y.dtype}`);
  }
  return [new ShapedArray(generalBroadcast(x.shape, y.shape), DType.Bool)];
}

function vectorizedUnopAbstractEval([x]: ShapedArray[]) {
  return [ShapedArray.fromAval(x)];
}

const abstractEvalRules: Record<Primitive, AbstractEvalRule> = {
  [Primitive.Add]: binopAbstractEval,
  [Primitive.Mul]: binopAbstractEval,
  [Primitive.Neg]: vectorizedUnopAbstractEval,
  [Primitive.Sin]: vectorizedUnopAbstractEval,
  [Primitive.Cos]: vectorizedUnopAbstractEval,
  [Primitive.ReduceSum]([x], { axis }: { axis: number[] }) {
    const axisSet = new Set(axis);
    const newShape = x.shape.filter((_, i) => !axisSet.has(i));
    return [new ShapedArray(newShape, x.dtype)];
  },
  [Primitive.Greater]: compareAbstractEval,
  [Primitive.Less]: compareAbstractEval,
  [Primitive.Transpose]([x], { perm }: { perm?: number[] }) {
    if (perm === undefined) {
      perm = [...JsArray(x.shape.length).keys()].reverse();
    }
    return [
      new ShapedArray(
        perm.map((i) => x.shape[i]),
        x.dtype
      ),
    ];
  },
  [Primitive.Broadcast](
    [x],
    { shape, axes }: { shape: number[]; axes: number[] }
  ) {
    return [new ShapedArray(shape, x.dtype)];
  },
};
