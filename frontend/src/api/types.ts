export interface AllocatorStats {
  total_blocks: number;
  block_size: number;
  free_blocks: number;
  allocated_blocks: number;
  utilization: number;
  request_block_tables: Record<string, number[]>;
}
export interface TokenEvent {
  request_id: string;
  token_id: number;
  token_index: number;
  token_text: string;
  generated_text: string;
  tick?: number;
  phase?: "prefill" | "decode";
}
export interface TraceEvent {
  serving_cache?: boolean;
  logical_cache_length_before?: number;
  logical_cache_length_after?: number;
  request_id: string;
  token_id: number;
  token_index?: number;
  token_text?: string;
  generated_text?: string;
  phase: "prefill" | "decode";
  logical_context_length: number;
  model_input_width: number;
  batch_size: number;
}
export interface TraceFrame {
  serving_cache?: boolean;
  prefill_request_ids?: string[];
  decode_request_ids?: string[];
  prefill_input_shape?: number[] | null;
  decode_input_shape?: number[] | null;
  model_forward_count_this_tick?: number;
  logical_cache_lengths_before?: Record<string, number>;
  logical_cache_lengths_after?: Record<string, number>;
  tick: number;
  admitted: string[];
  active_request_ids: string[];
  model_request_ids: string[];
  completed: string[];
  events: TraceEvent[];
  waiting_request_ids?: string[];
  queue_depth?: number;
  observed_at_ms?: number;
  worker_step_ms?: number;
  logical_kv_before_forward?: AllocatorStats;
  logical_kv_after_completion?: AllocatorStats;
}
export interface Snapshot {
  tick: number;
  active_request_ids: string[];
  waiting_request_ids: string[];
  queue_depth: number;
  batch_size: number;
  max_active: number;
  model_forward_count: number;
  in_flight: boolean;
  model: string;
  device: string;
  engine: string;
  engine_commit: string;
  serving_cache: boolean;
  logical_allocator: null;
  traces: TraceFrame[];
}
export interface GenerationResult {
  request_id: string;
  generated_token_ids: number[];
  generated_text: string;
  stop_reason: string;
  generated_token_count: number;
}
export type StreamEvent =
  | { event: "token"; data: TokenEvent }
  | { event: "done"; data: GenerationResult }
  | { event: "error"; data: { message: string } };
