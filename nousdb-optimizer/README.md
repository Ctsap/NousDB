# nousdb-optimizer

Part of [NousDB](https://crates.io/crates/nousdb), a semantic query engine on
Apache DataFusion. This crate contains the optimizer rules (semantic rewrites, histogram injection, token repartitioning, batch sizing, quality certification).

Most programs should depend on [`nousdb`](https://crates.io/crates/nousdb),
which re-exports what is needed from here.

Licensed under the Apache License, Version 2.0.
