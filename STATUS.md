# Backend continuation status

- Step 0: frozen artifact hash guard, exploratory report archive and disclosure; 1 focused test passed. Remaining: M10, M11 final one-shot run, M12 confirmation.
- Step B / M10: REDUCED, NOT COMPLETE. Amount splitting only at strengths 0 and 1.0; real validation graph, detector/model, and baseline reruns wrote `outputs/evasion.csv` and `.json`. Delays and decoys remain. `pytest -q`: 72 passed in 18.95s (elevated command). Main recall was 0.0 at both strengths; baseline attempt recall was 0.2632 then 1.0. Next: minimal M11 attempt recall, M12 fallback, then one final test report.
- Step D / M12: reduced offline-only fallback done; invalid LLM output restores the template and writes failure count/rate to `outputs/narrative_stats.json`. `pytest -q tests/planted/test_narrative.py`: 6 passed in 0.94s. No provider is configured.
