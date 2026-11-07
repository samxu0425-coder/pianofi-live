-- Where a pass spends its time (p50 ms per stage) for the real-time GPU runs with the cache on.
SELECT r.machine, r.stride_frames * 16 AS stride_ms, r.delay_frames * 16 AS delay_ms, r.name,
       MAX(CASE WHEN t.stage = 'frame_model_ms' THEN t.p50 END) AS frame_model,
       MAX(CASE WHEN t.stage = 'time_model_ms'  THEN t.p50 END) AS time_model,
       MAX(CASE WHEN t.stage = 'decode_ms'      THEN t.p50 END) AS decode,
       MAX(CASE WHEN t.stage = 'pass_ms'        THEN t.p50 END) AS pass,
       MAX(CASE WHEN t.stage = 'pass_ms'        THEN t.p99 END) AS pass_p99
FROM runs r
JOIN timings t ON t.run_id = r.run_id
WHERE r.provider = 'CUDA' AND r.cache = 1 AND r.realtime = 1 AND r.input = 'file'
GROUP BY r.run_id
ORDER BY r.machine, stride_ms, delay_ms;
