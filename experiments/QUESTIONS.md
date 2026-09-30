# Questions blocking experiment setup

- What exact metadata and image paths under `local/` should be configured for Camelyon16 and TCGA-NSCLC? The default `local/csv/*.csv` naming is inherited from the source and can be changed.
- Which source configuration, seeds, and result files reproduce MICCAI Tables 1 and 2? None were included in the clone.
- What backbone, patch size, magnification, and validation split should be used for the Phase 1 head-to-head?
- Should the incomplete DINO files be repaired now, or kept out of the initial Barlow Twins comparison? They lack the imported `src.ssl_methods.dino` implementation and the trainer's required multi-crop dataset.
- SimTriplet and SSLP are absent from the source; the plan requires faithful implementations and method notes before comparison.
