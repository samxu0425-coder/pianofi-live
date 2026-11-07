-- The same setting on every machine it ran on: does only speed change, or the output too?
SELECT stride_ms, delay_ms, gpu,
       COUNT(*) AS runs, MIN(f1) AS f1_min, MAX(f1) AS f1_max, MIN(notes) AS notes_min, MAX(notes) AS notes_max,
       MIN(pass_p50_ms) AS pass_p50_min, MAX(pass_p50_ms) AS pass_p50_max, SUM(overrun_samples) AS overruns
FROM run_summary
WHERE wav LIKE '%first60s%' AND provider = 'CUDA' AND cache = 1 AND heads = 'b' AND normalize = 0
  AND realtime = 1 AND f1 IS NOT NULL
GROUP BY stride_ms, delay_ms, gpu
HAVING COUNT(*) >= 1
ORDER BY stride_ms, delay_ms, gpu;
