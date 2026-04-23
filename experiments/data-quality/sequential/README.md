# data-quality/sequential

Surfaces probably-mis-labeled sequences in the pyro-dataset by running a
`pyrocore.TemporalModel` oracle over every sequence and presenting
disagreements against the folder-based ground truth as FiftyOne FP/FN
review sets.

See `../../../docs/specs/2026-04-23-sequential-label-audit-design.md` for design.
Reproduce steps are filled in at the end of the implementation.
