-- Stride 512 runs must reproduce batch exactly: F1 1.0, decoder ok, array check ok (where recorded).
SELECT r.name, r.machine, r.notes, s.f1, s.decoder_ok, s.array_ok, s.scored_at, s.source
FROM runs r
JOIN scores s ON s.run_id = r.run_id
WHERE r.stride_frames = 512 AND r.heads = 'b'
ORDER BY s.scored_at;
