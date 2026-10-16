import { readFile, readdir } from "node:fs/promises";
const paths = [];
async function walk(dir) {
  for (const item of await readdir(dir, { withFileTypes: true })) {
    const path = `${dir}/${item.name}`;
    if (item.isDirectory()) await walk(path);
    else if (/\.(css|tsx)$/.test(path) && !path.includes("/test/"))
      paths.push(path);
  }
}
await walk("src");
const failures = [];
for (const path of paths) {
  const source = await readFile(path, "utf8");
  for (const forbidden of [
    /\b(?:linear|radial|conic|repeating-linear)-gradient\s*\(/i,
    /backdrop-filter/i,
    /\bblur\s*\(/i,
    /font-weight\s*:\s*[6-9]00/i,
    /\b(?:Card|StatCard|GradientPanel)\b/,
  ])
    if (forbidden.test(source)) failures.push(`${path}: ${forbidden}`);
  if (path.endsWith(".css")) {
    for (const match of source.matchAll(/border-radius\s*:\s*([^;}]+)/g)) {
      const sizes = match[1].match(/\d+(?:\.\d+)?(?:px|rem|em|%)/g) ?? [];
      if (sizes.some((n) => !n.endsWith("px") || parseFloat(n) > 4))
        failures.push(`${path}: excessive radius ${match[1]}`);
    }
    for (const match of source.matchAll(/box-shadow\s*:\s*([^;}]+)/g))
      if (
        match[1].trim() !== "none" &&
        match[1].trim() !== "0 8px 24px rgb(0 0 0 / 0.45)"
      )
        failures.push(`${path}: unapproved shadow`);
    const approved = new Set([
      "#0e0f10",
      "#151618",
      "#1b1c1f",
      "#25272b",
      "#33363b",
      "#1f2124",
      "#e8e6e1",
      "#a19f98",
      "#6f6e69",
      "#efa23a",
      "#f5b65c",
      "#1a1307",
      "#cfcbc2",
      "#55575d",
      "#5cb98b",
      "#d8c34a",
      "#e35f4c",
    ]);
    for (const color of source.matchAll(/#[a-f\d]{6}\b/gi))
      if (!approved.has(color[0].toLowerCase()))
        failures.push(`${path}: unapproved color ${color[0]}`);
  }
}
const pkg = JSON.parse(await readFile("package.json", "utf8"));
for (const name of Object.keys({ ...pkg.dependencies, ...pkg.devDependencies }))
  if (
    /lucide|heroicons|fontawesome|phosphor|material-icons|tailwind|shadcn|chakra|mantine|antd|tremor|bootstrap|recharts|chart\.js|echarts|apexcharts|^@mui\//i.test(
      name,
    )
  )
    failures.push(`Forbidden UI dependency: ${name}`);
if (failures.length) {
  console.error(failures.join("\n"));
  process.exitCode = 1;
} else
  console.log(
    `Trace static design audit passed: ${paths.length} visual source files; approved palette, weights, radii, shadows; no decorative UI/icon/chart dependency.`,
  );
