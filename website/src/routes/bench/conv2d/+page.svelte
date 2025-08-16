<script lang="ts">
  import { browser } from "$app/environment";

  import type { Tensor4D } from "@tensorflow/tfjs";

  const batchSize = 1;
  const channels = 32;
  const height = 256;
  const width = 256;
  const filterHeight = 3;
  const filterWidth = 3;
  const outChannels = 64;

  let result: Record<string, number> = $state({});

  const inputSize = batchSize * channels * height * width;
  const filterSize = outChannels * channels * filterHeight * filterWidth;
  const outputSize = batchSize * outChannels * height * width; // assuming same padding

  const randomInput = new Float32Array(
    [...new Array(inputSize)].map(() => Math.random()),
  );
  const randomFilter = new Float32Array(
    [...new Array(filterSize)].map(() => Math.random()),
  );

  function printBufferItems(buf: Float32Array) {
    console.log(
      buf[0],
      buf[1],
      buf[2],
      buf[3],
      buf[Math.floor(buf.length / 2)],
      buf[buf.length - 1],
    );
  }

  abstract class Strategy {
    abstract name: string;
    abstract run(): Promise<number>;
  }

  abstract class GpuStrategy extends Strategy {
    abstract kernel(): string;
    abstract workgroups(): [number, number, number];

    async run(): Promise<number> {
      const adapter = await navigator.gpu.requestAdapter({
        powerPreference: "high-performance",
      });
      if (!adapter) {
        alert("WebGPU not supported");
        return -1;
      }

      let device: GPUDevice;
      try {
        device = await adapter.requestDevice({
          requiredLimits: {
            maxComputeInvocationsPerWorkgroup:
              adapter.limits.maxComputeInvocationsPerWorkgroup,
            maxComputeWorkgroupSizeX: adapter.limits.maxComputeWorkgroupSizeX,
            maxComputeWorkgroupSizeY: adapter.limits.maxComputeWorkgroupSizeY,
            maxComputeWorkgroupSizeZ: adapter.limits.maxComputeWorkgroupSizeZ,
            maxComputeWorkgroupStorageSize:
              adapter.limits.maxComputeWorkgroupStorageSize,
            maxComputeWorkgroupsPerDimension:
              adapter.limits.maxComputeWorkgroupsPerDimension,
            maxStorageBufferBindingSize:
              adapter.limits.maxStorageBufferBindingSize,
          },
        });
      } catch (error) {
        console.warn("Error when creating device:", error);
        return -1;
      }

      const usage =
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST;
      const input = device.createBuffer({ size: inputSize * 4, usage });
      const filter = device.createBuffer({ size: filterSize * 4, usage });
      const output = device.createBuffer({ size: outputSize * 4, usage });
      const staging = device.createBuffer({
        size: outputSize * 4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });

      device.queue.writeBuffer(input, 0, randomInput);
      device.queue.writeBuffer(filter, 0, randomFilter);

      try {
        const pipeline = await device.createComputePipelineAsync({
          compute: {
            module: device.createShaderModule({ code: this.kernel() }),
            entryPoint: "main",
          },
          layout: "auto",
        });

        const bindGroup = device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: input } },
            { binding: 1, resource: { buffer: filter } },
            { binding: 2, resource: { buffer: output } },
          ],
        });

        const commandEncoder = device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(pipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(...this.workgroups());
        passEncoder.end();
        commandEncoder.copyBufferToBuffer(
          output,
          0,
          staging,
          0,
          outputSize * 4,
        );
        device.queue.submit([commandEncoder.finish()]);

        const start = performance.now();

        await staging.mapAsync(GPUMapMode.READ, 0, outputSize * 4);
        const buf = new Float32Array(staging.getMappedRange());
        printBufferItems(buf);
        staging.unmap();

        return (performance.now() - start) / 1000;
      } finally {
        input.destroy();
        filter.destroy();
        output.destroy();
        staging.destroy();
      }
    }
  }

  class NaiveStrategy extends GpuStrategy {
    name: string;
    blocksize: number;

    constructor(block: number) {
      super();
      this.name = `naive-${block}`;
      this.blocksize = block;
    }

    kernel() {
      return `
@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read> weights : array<f32>;
@group(0) @binding(2) var<storage, read_write> output : array<f32>;

const BATCH_SIZE: u32 = ${batchSize}u;
const IN_CHANNELS: u32 = ${channels}u;
const HEIGHT: u32 = ${height}u;
const WIDTH: u32 = ${width}u;
const FILTER_HEIGHT: u32 = ${filterHeight}u;
const FILTER_WIDTH: u32 = ${filterWidth}u;
const OUT_CHANNELS: u32 = ${outChannels}u;

fn input_idx(b: u32, c: u32, h: u32, w: u32) -> u32 {
  return b * IN_CHANNELS * HEIGHT * WIDTH + c * HEIGHT * WIDTH + h * WIDTH + w;
}

fn weights_idx(oc: u32, ic: u32, fh: u32, fw: u32) -> u32 {
  return oc * IN_CHANNELS * FILTER_HEIGHT * FILTER_WIDTH + ic * FILTER_HEIGHT * FILTER_WIDTH + fh * FILTER_WIDTH + fw;
}

fn output_idx(b: u32, oc: u32, h: u32, w: u32) -> u32 {
  return b * OUT_CHANNELS * HEIGHT * WIDTH + oc * HEIGHT * WIDTH + h * WIDTH + w;
}

@compute @workgroup_size(${this.blocksize}, ${this.blocksize}, 1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
  let out_h: u32 = global_id.y;
  let out_w: u32 = global_id.x;
  let out_c: u32 = global_id.z;

  if (out_h >= HEIGHT || out_w >= WIDTH || out_c >= OUT_CHANNELS) {
    return;
  }

  for (var b: u32 = 0u; b < BATCH_SIZE; b = b + 1u) {
    var sum: f32 = 0.0;

    for (var ic: u32 = 0u; ic < IN_CHANNELS; ic = ic + 1u) {
      for (var fh: u32 = 0u; fh < FILTER_HEIGHT; fh = fh + 1u) {
        for (var fw: u32 = 0u; fw < FILTER_WIDTH; fw = fw + 1u) {
          let in_h: i32 = i32(out_h) + i32(fh) - i32(FILTER_HEIGHT / 2u);
          let in_w: i32 = i32(out_w) + i32(fw) - i32(FILTER_WIDTH / 2u);

          if (in_h >= 0 && in_h < i32(HEIGHT) && in_w >= 0 && in_w < i32(WIDTH)) {
            let input_val: f32 = input[input_idx(b, ic, u32(in_h), u32(in_w))];
            let weights_val: f32 = weights[weights_idx(out_c, ic, fh, fw)];
            sum = sum + input_val * weights_val;
          }
        }
      }
    }

    output[output_idx(b, out_c, out_h, out_w)] = sum;
  }
}
`;
    }

    workgroups(): [number, number, number] {
      return [
        Math.ceil(width / this.blocksize),
        Math.ceil(height / this.blocksize),
        outChannels,
      ];
    }
  }

  class TfjsStrategy extends Strategy {
    name = "tfjs";

    async run(): Promise<number> {
      const tf = await import("@tensorflow/tfjs");
      await import("@tensorflow/tfjs-backend-webgpu");
      await tf.setBackend("webgpu");

      // Use shared random data with NCHW format for input.
      //
      // However, even though tfjs has a "NCHW" format in their documentation,
      // it appears to produce generate invalid kernels in their WebGPU backend
      // as the output is wrong. Probably just a bug in the tfjs-backend-webgpu,
      // since it works in tfjs-backend-webgl (but that is much slower).
      //
      // That's not an issue though, since we can just transpose the input and
      // output in lieu of debugging tfjs to find a fix.
      const input = tf
        .tensor4d(randomInput, [batchSize, channels, height, width])
        .transpose<Tensor4D>([0, 2, 3, 1]); // NHWC format

      // Convert filter from OIHW to HWIO format using transpose
      const filterOIHW = tf.tensor4d(randomFilter, [
        outChannels,
        channels,
        filterHeight,
        filterWidth,
      ]);
      const filter = tf.transpose(filterOIHW, [2, 3, 1, 0]); // OIHW -> HWIO
      await Promise.all([input.data(), filter.data()]);

      const start = performance.now();
      const output = tf
        .conv2d(input, filter, 1, "same", "NHWC")
        .transpose<Tensor4D>([0, 3, 1, 2]); // NHWC -> NCHW
      const ar = (await output.data()) as Float32Array;
      printBufferItems(ar);
      const time = performance.now() - start;

      input.dispose();
      filterOIHW.dispose();
      filter.dispose();
      output.dispose();

      return time / 1000;
    }
  }

  class JaxJsStrategy extends Strategy {
    name = "jax-js";

    async run(): Promise<number> {
      const jax = await import("@jax-js/jax");
      await jax.init();
      jax.setDevice("webgpu");
      const np = jax.numpy;

      const input = np.array(randomInput, {
        shape: [batchSize, channels, height, width],
      });
      const filter = np.array(randomFilter, {
        shape: [outChannels, channels, filterHeight, filterWidth],
      });
      await Promise.all([input.ref.wait(), filter.ref.wait()]);

      const start = performance.now();
      const output = jax.lax.convGeneralDilated(
        input,
        filter,
        [1, 1], // strides
        [
          [1, 1],
          [1, 1],
        ], // padding
      );
      const ar = (await output.data()) as Float32Array;
      printBufferItems(ar);
      const time = performance.now() - start;

      return time / 1000;
    }
  }

  const strategiesList: Strategy[] = [
    new NaiveStrategy(8),
    new NaiveStrategy(16),
    new NaiveStrategy(32),
    new TfjsStrategy(),
    new JaxJsStrategy(),
  ];

  const strategies = Object.fromEntries(strategiesList.map((s) => [s.name, s]));

  async function bench(variant: string) {
    console.log(`Running ${variant}...`);
    const time = await strategies[variant].run();
    if (time >= 0) {
      result[variant] = time;
    } else {
      console.error(`Error running ${variant}`);
    }
  }
</script>

<main class="p-4">
  <h1 class="text-2xl mb-2">conv2d benchmark</h1>

  <p class="mb-4">
    Running different WebGPU conv2d implementations on {batchSize}x{channels}x{height}x{width}
    input with {outChannels} filters of size {filterHeight}x{filterWidth}.
  </p>

  <div class="flex flex-wrap gap-2 mb-4">
    {#each strategiesList as strategy (strategy.name)}
      <button
        class="border px-2 hover:bg-gray-100 active:scale-95"
        onclick={() => bench(strategy.name)}
      >
        {strategy.name}
      </button>
    {/each}
  </div>

  {#if browser && !navigator.gpu}
    <p class="text-red-500 mb-4">
      WebGPU is not supported. Benchmarks will not work.
    </p>
  {/if}

  {#each Object.entries(result) as [variant, time]}
    <div>
      <span class="font-bold">{variant}:</span>
      {time.toFixed(3)} seconds,
      {(
        (2 *
          batchSize *
          outChannels *
          channels *
          height *
          width *
          filterHeight *
          filterWidth) /
        1e9 /
        time
      ).toFixed(2)} GFLOP/s
    </div>
  {/each}
</main>

<style lang="postcss">
  @reference "$app.css";

  button {
    @apply border rounded px-2 hover:bg-gray-100 active:scale-95;
  }
</style>
