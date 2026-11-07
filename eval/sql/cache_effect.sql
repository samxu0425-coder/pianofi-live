-- Per-frame cache on vs off: same machine, clip, provider and stride. Speedup = pass p50 off / on.
SELECT on_.gpu, on_.provider, on_.stride_ms,
       off.pass_p50_ms AS pass_p50_off, on_.pass_p50_ms AS pass_p50_on,
       ROUND(off.pass_p50_ms / on_.pass_p50_ms, 2) AS speedup,
       off.gpu_memory_mb AS gpu_mb_off, on_.gpu_memory_mb AS gpu_mb_on,
       off.overrun_samples AS overruns_off, on_.overrun_samples AS overruns_on,
       off.f1 AS f1_off, on_.f1 AS f1_on
FROM run_summary on_
JOIN run_summary off
  ON off.machine = on_.machine AND off.wav = on_.wav AND off.provider = on_.provider
 AND off.stride_ms = on_.stride_ms AND off.delay_ms = on_.delay_ms AND off.heads = on_.heads
 AND off.cache = 0 AND on_.cache = 1 AND off.realtime = on_.realtime
ORDER BY on_.gpu, on_.provider, on_.stride_ms;
