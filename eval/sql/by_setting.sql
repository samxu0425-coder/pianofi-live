-- F1, speed and lag per setting on the 60 s clip, latest score per run (GPU, no auto level).
SELECT gpu, stride_ms, delay_ms, heads, cache,
       f1, pass_p50_ms, pass_p99_ms, lag_p50_ms, lag_p99_ms, overrun_samples, gpu_memory_mb, name
FROM run_summary
WHERE wav LIKE '%first60s%' AND provider = 'CUDA' AND normalize = 0 AND f1 IS NOT NULL
ORDER BY gpu, stride_ms, delay_ms, cache DESC, heads;
