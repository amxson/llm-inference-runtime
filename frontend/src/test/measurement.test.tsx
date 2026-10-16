import { render, screen } from "@testing-library/react";
import { readFileSync } from "node:fs";
import { FinalMeasurement, meanByPosition } from "../views/FinalMeasurement";
import type { Artifact } from "../data/benchmarkAdapter";

test("position means use raw observations, not fabricated trend points", () => {
  expect(
    meanByPosition([
      [1, 10, 3],
      [3, 20, 7],
    ]),
  ).toEqual([2, 15, 5]);
  expect(meanByPosition([])).toEqual([]);
});
test("final measurement renders real artifact, distinct timing scopes and host limits", () => {
  const data = JSON.parse(
    readFileSync("../benchmarks/results/final_cpu.json", "utf8"),
  );
  render(
    <FinalMeasurement
      artifact={
        {
          name: "final_cpu.json",
          source: "benchmarks/results/final_cpu.json",
          source_sha256: "test",
          source_commit: "test",
          data,
        } as Artifact
      }
    />,
  );
  expect(
    screen.getByRole("img", { name: /local token selection/ }),
  ).toBeInTheDocument();
  expect(
    screen.getByRole("img", { name: /HTTP SSE arrival/ }),
  ).toBeInTheDocument();
  expect(screen.getByText(/not an isolated-host baseline/)).toBeInTheDocument();
  expect(screen.getByText(/not proven causality/)).toBeInTheDocument();
  expect(screen.getByRole("link", { name: "Final raw JSON" })).toHaveAttribute(
    "href",
    "/data/final_cpu.json",
  );
});
