import { AluExp, AluOp, DType } from "./alu";
import {
  accessorAluExp,
  accessorGlobal,
  Backend,
  BackendType,
  Executable,
  getBackend,
  Slot,
} from "./backend";
import { ShapeTracker } from "./shape";
import { deepEqual } from "./utils";

const JsArray = globalThis.Array;

class PendingExecute {
  prepared: Executable | null = null;
  submitted = false;
  #promise: Promise<void> | null = null; // for prepare

  constructor(
    readonly exp: AluExp,
    readonly inputs: Slot[],
    readonly outputs: Slot[],
  ) {}

  async prepare(backend: Backend) {
    if (this.prepared) return;
    if (this.#promise) {
      await this.#promise;
      return;
    }
    this.#promise = (async () => {
      this.prepared = await backend.prepare(this.inputs.length, this.exp);
    })();
    await this.#promise;
  }

  prepareSync(backend: Backend) {
    if (this.prepared) return;
    this.prepared = backend.prepareSync(this.inputs.length, this.exp);
  }

  submit(backend: Backend) {
    if (this.submitted) return;
    if (!this.prepared) throw new Error("Not prepared yet");
    backend.dispatch(this.prepared, this.inputs, this.outputs);
  }
}

export class Array {
  readonly shape: number[];
  readonly dtype: DType;

  #source: AluExp | Slot;
  #st: ShapeTracker;
  #backend: Backend;
  #pending: Set<PendingExecute> | null; // only if source is `Slot`

  constructor(
    source: AluExp | Slot,
    st: ShapeTracker,
    dtype: DType,
    backend: Backend,
    pending: Set<PendingExecute> | null = null,
  ) {
    this.shape = st.shape;
    this.dtype = dtype;

    this.#source = source;
    this.#st = st;
    this.#backend = backend;
    this.#pending = pending;
  }

  get backend(): BackendType {
    return this.#backend.type;
  }

  static zeros(
    shape: number[],
    { dtype, backend }: { dtype?: DType; backend?: BackendType } = {},
  ) {
    dtype = dtype ?? DType.Float32;
    return new Array(
      AluExp.const(dtype, 0),
      ShapeTracker.fromShape(shape),
      dtype,
      getBackend(backend),
    );
  }

  static ones(
    shape: number[],
    { dtype, backend }: { dtype?: DType; backend?: BackendType } = {},
  ) {
    dtype = dtype ?? DType.Float32;
    return new Array(
      AluExp.const(dtype, 1),
      ShapeTracker.fromShape(shape),
      dtype,
      getBackend(backend),
    );
  }

  #binary(op: AluOp, other: Array) {
    if (!deepEqual(this.shape, other.shape)) {
      throw new Error(`Shape mismatch in ${op}`); // todo: broadcasting, maybe at the jax level
    }
    if (this.dtype !== other.dtype) {
      throw new Error(`Dtype mismatch in ${op}`); // todo: dtype casting
    }

    // Short circuit if both are already AluExp.
    if (this.#source instanceof AluExp && other.#source instanceof AluExp) {
      const exp = new AluExp(op, this.dtype, [this.#source, other.#source]);
      return new Array(exp, this.#st, this.dtype, this.#backend);
    }

    const gidx = AluExp.special(DType.Int32, "gidx", this.#st.size);

    const inputs: Slot[] = [];
    const src: AluExp[] = [];

    for (const ar of [this, other]) {
      if (ar.#source instanceof AluExp) {
        src.push(accessorAluExp(ar.#source, ar.#st, gidx));
      } else {
        src.push(accessorGlobal(inputs.length, ar.#st, gidx));
        inputs.push(ar.#source);
      }
    }

    const exp = new AluExp(op, this.dtype, src);
    const output = this.#backend.malloc(this.#st.size * 4);
    const pending = new Set([
      ...(this.#pending ?? []),
      ...(other.#pending ?? []),
      new PendingExecute(exp, inputs, [output]),
    ]);
    return new Array(
      output,
      ShapeTracker.fromShape(this.shape),
      this.dtype,
      this.#backend,
      pending,
    );
  }

  /** Normalizes this array into one backed by a `Slot`. */
  #toSlot(): Array {
    if (!(this.#source instanceof AluExp)) return this;
    const output = this.#backend.malloc(this.#st.size * 4);
    const gidx = AluExp.special(DType.Int32, "gidx", this.#st.size);
    const exp = accessorAluExp(this.#source, this.#st, gidx);
    const pending = new Set([new PendingExecute(exp, [], [output])]);
    return new Array(
      output,
      ShapeTracker.fromShape(this.shape),
      this.dtype,
      this.#backend,
      pending,
    );
  }

  // These will be evaluation rules in the future, not public API.
  add(other: Array) {
    return this.#binary(AluOp.Add, other);
  }
  sub(other: Array) {
    return this.#binary(AluOp.Sub, other);
  }
  mul(other: Array) {
    return this.#binary(AluOp.Mul, other);
  }

  async data(): Promise<Float32Array> {
    const array = this.#toSlot();
    if (array.#pending) {
      // Compile all pending executables concurrently.
      await Promise.all(
        [...array.#pending].map((exe) => exe.prepare(this.#backend)),
      );
      for (const p of array.#pending) {
        p.submit(this.#backend);
      }
    }
    const buf = await this.#backend.read(array.#source as Slot);
    return new Float32Array(buf);
  }

  dataSync(): Float32Array {
    const array = this.#toSlot();
    if (array.#pending) {
      for (const p of array.#pending) {
        p.prepareSync(this.#backend);
        p.submit(this.#backend);
      }
    }
    const buf = this.#backend.readSync(array.#source as Slot);
    return new Float32Array(buf);
  }
}
