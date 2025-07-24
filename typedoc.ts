// Executable script for generating typedoc documentation.

import { spawn } from "node:child_process";
import { readdir } from "node:fs/promises";

/** Run a shell command, echo its output, and throw on non‑zero exit. */
async function sh(cmd: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const child = spawn(cmd, { shell: true, stdio: "inherit" });
    child.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(`Command failed with exit code ${code}: ${cmd}`));
      } else {
        resolve();
      }
    });
  });
}

// Docs for @jax-js/jax.
await sh(`pnpm typedoc src/index.ts --json docs-json/jax.json`);

// Generate docs for each package in the packages directory.
for (const pkg of await readdir("packages")) {
  await sh(
    `pnpm typedoc packages/${pkg}/src/index.ts --json docs-json/${pkg}.json`,
  );
}

// Merge all package docs into a single HTML output.
await sh(
  `pnpm typedoc --name jax-js --entryPointStrategy merge "docs-json/*.json"`,
);
