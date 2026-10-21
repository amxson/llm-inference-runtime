// Derive offline artifacts from tracked sources. Never duplicate benchmark numbers by hand.
import { readFile, mkdir, writeFile } from "node:fs/promises";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { fileURLToPath } from "node:url";
import path from "node:path";
const root = fileURLToPath(new URL("../../", import.meta.url));
let manifest = [];
try {
  manifest = JSON.parse(
    await readFile(path.join(root, "benchmarks/provenance.json"), "utf8"),
  );
} catch (error) {
  if (error.code !== "ENOENT") throw error;
}
let files;
try {
  files = execFileSync("git", ["ls-files", "benchmarks/results/*.json"], {
    cwd: root,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
  })
    .trim()
    .split(/\r?\n/)
    .filter(Boolean);
} catch {
  files = manifest.map((item) => item.source);
}
if (!files.length) files = manifest.map((item) => item.source);
if (!files.length)
  throw new Error(
    "No tracked artifacts or checked provenance manifest available.",
  );
const artifacts = await Promise.all(
  files.map(async (source) => {
    const text = await readFile(path.join(root, source), "utf8");
    const hash = createHash("sha256")
      .update(text.replace(/\r\n/g, "\n"))
      .digest("hex");
    let commit;
    try {
      commit = execFileSync("git", ["log", "-1", "--format=%H", "--", source], {
        cwd: root,
        encoding: "utf8",
        stdio: ["ignore", "pipe", "pipe"],
      }).trim();
    } catch { /* Git metadata may be absent in uploaded source bundles. */ }
    if (!commit) {
      const entry = manifest.find((item) => item.source === source);
      if (!entry || entry.source_sha256 !== hash)
        throw new Error(`Missing/stale provenance: ${source}`);
      commit = entry.source_commit;
    }
    return {
      name: path.basename(source),
      source,
      source_commit: commit,
      source_sha256: hash,
      data: JSON.parse(text),
    };
  }),
);
if (process.argv.includes("--write-manifest")) {
  if (artifacts.some((a) => !a.source_commit))
    throw new Error("Commit artifacts before recording deployment provenance.");
  await writeFile(
    path.join(root, "benchmarks/provenance.json"),
    JSON.stringify(
      artifacts.map(({ name, source, source_commit, source_sha256 }) => ({
        name,
        source,
        source_commit,
        source_sha256,
      })),
      null,
      2,
    ) + "\n",
  );
}
const out = path.join(root, "frontend/public/data");
await mkdir(out, { recursive: true });
await writeFile(
  path.join(out, "benchmarks.json"),
  JSON.stringify({ artifacts: artifacts.map((artifact) => {
    if (artifact.name !== "gpu_final_comparison.json") return artifact;
    // Keep raw requests/events/forward traces in the linked, unchanged JSON.
    // The page needs aggregates, not many copies of the full GPU recording.
    const { raw_repetitions, warmups, source_file_sha256, sessions, ...data } = artifact.data;
    return { ...artifact, data: {
      ...data,
      presentation_projection: "GPU summary; full source available through Raw JSON",
      measured_cohorts: raw_repetitions.length,
      excluded_warmup_cohorts: warmups.length,
      sessions: sessions.map(({ index, engine, command, environment }) => ({ index, engine, command, environment })),
      aggregates: data.aggregates.map(({ comparison, ...row }) => {
        const { outputs, ...summary } = comparison;
        return { ...row, comparison: summary };
      }),
    } };
  }) }),
);
for (const artifact of artifacts)
  await writeFile(
    path.join(out, artifact.name),
    await readFile(path.join(root, artifact.source), "utf8"),
  );
console.log(
  `Derived ${artifacts.length} benchmark artifacts from Git-tracked JSON`,
);
