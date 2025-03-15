export enum DType {
  Float32 = "float32",
  Int32 = "int32",
  Bool = "bool",
  Complex64 = "complex64",
}

/**
 * Mathemtical expression on scalar values.
 *
 * This is similiar to and based on tinygrad's UOp class, but it's more specific
 * to just math on scalars. We're doing this to avoid the complexity of a full
 * graph rewrite engine.
 */
export class AluExp {
  constructor(
    readonly op: AluOp,
    readonly dtype: DType,
    readonly src: AluExp[],
    readonly arg: any = undefined,
  ) {}

  static add(a: AluExp, b: AluExp): AluExp {
    return new AluExp(AluOp.Add, a.dtype, [a, b]);
  }
  static sub(a: AluExp, b: AluExp): AluExp {
    return new AluExp(AluOp.Sub, a.dtype, [a, b]);
  }
  static mul(a: AluExp, b: AluExp): AluExp {
    return new AluExp(AluOp.Mul, a.dtype, [a, b]);
  }
  static idiv(a: AluExp, b: AluExp): AluExp {
    return new AluExp(AluOp.Idiv, a.dtype, [a, b]);
  }
  static mod(a: AluExp, b: AluExp): AluExp {
    return new AluExp(AluOp.Mod, a.dtype, [a, b]);
  }
  static cmplt(a: AluExp, b: AluExp): AluExp {
    return new AluExp(AluOp.Cmplt, DType.Bool, [a, b]);
  }
  static cmpne(a: AluExp, b: AluExp): AluExp {
    return new AluExp(AluOp.Cmpne, DType.Bool, [a, b]);
  }
  static where(cond: AluExp, a: AluExp, b: AluExp): AluExp {
    return new AluExp(AluOp.Where, a.dtype, [cond, a, b]);
  }
  static const(dtype: DType, value: any): AluExp {
    return new AluExp(AluOp.Const, dtype, [], value);
  }
  static special(dtype: DType, name: string, n: number): AluExp {
    return new AluExp(AluOp.Special, dtype, [], [name, n]);
  }

  static i32(value: number): AluExp {
    return AluExp.const(DType.Int32, value);
  }
  static f32(value: number): AluExp {
    return AluExp.const(DType.Float32, value);
  }
  static bool(value: boolean): AluExp {
    return AluExp.const(DType.Bool, value);
  }

  not(): AluExp {
    if (this.dtype !== DType.Bool) {
      throw new Error("not() can only be called on boolean expressions");
    }
    return AluExp.cmpne(this, AluExp.const(DType.Bool, true));
  }

  /** Resolve this to a value, or `undefined` if not possible. */
  resolve(): any | undefined {
    if (this.op === AluOp.Const) return this.arg;
    return undefined;
  }

  /** Evaluate the expression, returning the result (or a list of results). */
  evaluate(context: Record<string, any>): any {
    let x, y;
    switch (this.op) {
      case AluOp.Add:
        x = this.src[0].evaluate(context);
        y = this.src[1].evaluate(context);
        return this.dtype === DType.Bool ? x || y : x + y;
      case AluOp.Sub:
        return this.src[0].evaluate(context) - this.src[1].evaluate(context);
      case AluOp.Mul:
        x = this.src[0].evaluate(context);
        y = this.src[1].evaluate(context);
        return this.dtype === DType.Bool ? x && y : x * y;
      case AluOp.Idiv:
        return Math.floor(
          this.src[0].evaluate(context) / this.src[1].evaluate(context),
        );
      case AluOp.Mod:
        return this.src[0].evaluate(context) % this.src[1].evaluate(context);
      case AluOp.Cmplt:
        return this.src[0].evaluate(context) < this.src[1].evaluate(context);
      case AluOp.Cmpne:
        return this.src[0].evaluate(context) != this.src[1].evaluate(context);
      case AluOp.Where:
        return this.src[0].evaluate(context)
          ? this.src[1].evaluate(context)
          : this.src[2].evaluate(context);
      case AluOp.Const:
        return this.arg;
      case AluOp.Special:
        return context[this.arg[0]];
    }
  }
}

/** Symbolic form for each mathematical operation. */
export enum AluOp {
  Add,
  Sub,
  Mul,
  Idiv,
  Mod,
  Cmplt,
  Cmpne,
  Where,
  Const, // arg = value
  Special, // arg = [variable_name, n]
}
