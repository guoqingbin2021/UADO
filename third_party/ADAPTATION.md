# Formal baseline and SOTA adaptation boundary

The five comparison repositories have intended upstream commit identifiers in
`SOURCES.json`. MAPPO-adapted imports the official `R_MAPPOPolicy` actor and
critic modules, while HAPPO-adapted imports official HAPPO actor modules and
organizes them into fixed local/RSU/UAV/defer update groups. Both neural rows
use this project's common constrained PPO/GAE loss; the upstream `R_MAPPO`
trainer is not instantiated, and the HAPPO adapter does not claim the upstream
`factor_batch` correction. AMCoEdge, FDEdge, and MEC-UARA retain the components
listed in the manifest.

Every method is connected to the same fixed-candidate DAG interface, base
physical/queue mask, event-driven environment, and deterministic lower-layer
allocator. UAMCO-DAG variants additionally apply their registered high-fan-in
causal post-filter and deterministic controllers. The package therefore
provides executable common-environment adaptations, not bit-for-bit
reproductions or identical-action-filter comparisons. No upstream result file
is imported into the UAMCO-DAG evaluation.

The commit strings for MAPPO and HAPPO are provenance declarations rather than
byte-identity certificates: their local directories contain no nested Git
metadata and `SOURCES.json` has no per-file SHA-256 entries for those two
adapters. The three domain adapters do have the audited hashes recorded in the
manifest. The historical `preserved_core` text in that frozen manifest records
the intended adapter family and update ordering; it must not be read as a claim
that the upstream MAPPO/HAPPO trainers are executed. A missing source, invalid
score shape, non-finite update, or missing
real dataset stops the job instead of silently substituting a heuristic.

All methods use the same:

- WfCommons Montage, Seismology, and Cycles workflow records;
- RELLIS-3D training/test mobility and M2DGR-Outdoor zero-shot mobility;
- actual predecessor-output delivery, byte-level pause/resume/restart semantics;
- finite transfer and compute queues;
- fixed candidate selection and base executor action mask;
- SLA-aware deterministic CPU and bandwidth allocation;
- 300 training episodes, seven paired seeds, three leave-one-family-out folds,
  and the same evaluation trajectories.

MAPPO and HAPPO are redistributed under MIT licenses, and MEC-UARA under Apache
License 2.0. AMCoEdge and FDEdge do not declare repository licenses; their
provenance and audited hashes are recorded without asserting redistribution
rights.
