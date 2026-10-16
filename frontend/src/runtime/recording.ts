import fixture from "../data/recorded-trace.json";
import { parseTraceFrames } from "./traceAdapter";
export const recordedFrames = parseTraceFrames(fixture.traces);
