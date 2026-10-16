// Derive offline artifacts from tracked sources. Never duplicate benchmark numbers by hand.
import { readFile, mkdir, writeFile } from "node:fs/promises";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { fileURLToPath } from "node:url";
import path from "node:path";
const root = fileURLToPath(new URL("../../", import.meta.url));
const files = execFileSync("git", ["ls-files", "benchmarks/results/*.json"], {
  cwd: root,
  encoding: "utf8",
})
  .trim()
  .split(/\r?\n/)
  .filter(Boolean);
const artifacts = await Promise.all(
  files.map(async (source) => {
    const text = await readFile(path.join(root, source), "utf8");
    return {
      name: path.basename(source),
      source,
      source_commit: execFileSync(
        "git",
        ["log", "-1", "--format=%H", "--", source],
        { cwd: root, encoding: "utf8" },
      ).trim(),
      source_sha256: createHash("sha256")
        .update(text.replace(/\r\n/g, "\n"))
        .digest("hex"),
      data: JSON.parse(text),
    };
  }),
);
const out = path.join(root, "frontend/public/data");
await mkdir(out, { recursive: true });
await writeFile(
  path.join(out, "benchmarks.json"),
  JSON.stringify({ artifacts }),
);
for (const artifact of artifacts)
  await writeFile(
    path.join(out, artifact.name),
    await readFile(path.join(root, artifact.source), "utf8"),
  );
console.log(
  `Derived ${artifacts.length} benchmark artifacts from Git-tracked JSON`,
);
