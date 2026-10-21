// Mechanical Markdown -> output-free notebook mirror; never executes cells.
import { readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = fileURLToPath(new URL("../", import.meta.url));
for (const [document, notebook] of [
  ["docs/phase3-colab.md", "notebooks/phase3_colab.ipynb"],
  ["docs/phase3-colab-one-cell.md", "notebooks/phase3_colab_one_cell.ipynb"],
]) {
  const text = (await readFile(path.join(root, document), "utf8")).replace(/\r\n/g, "\n");
  const cells = [];
  let cursor = 0;
  function append(kind, source) {
    if (!source.trim()) return;
    cells.push({ cell_type: kind, id: `phase3-${cells.length}`, metadata: {},
      ...(kind === "code" ? { execution_count: null, outputs: [] } : {}),
      source: source.match(/.*\n|.+$/g) ?? [],
    });
  }
  for (const match of text.matchAll(/^```python\n([\s\S]*?)^```/gm)) {
    append("markdown", text.slice(cursor, match.index));
    append("code", match[1]);
    cursor = match.index + match[0].length;
  }
  append("markdown", text.slice(cursor));
  const output = JSON.stringify({ cells, metadata: {
    kernelspec: { display_name: "Python 3", language: "python", name: "python3" },
    language_info: { name: "python" },
  }, nbformat: 4, nbformat_minor: 5 }, null, 2) + "\n";
  if (process.argv.includes("--write")) await writeFile(path.join(root, notebook), output);
  else if ((await readFile(path.join(root, notebook), "utf8")).replace(/\r\n/g, "\n") !== output)
    throw new Error(`Notebook mirror differs: ${notebook}; run this script with --write`);
  console.log(`Notebook mirror checked: ${notebook}`);
}
