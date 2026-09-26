# Reproduction layout

The numerical experiment was run with Python 3.12, NumPy 2.3.5, SciPy 1.18.1 and PyTorch 2.14.0 on a CPU, with one Torch thread per training process. The exact installed version list is saved separately in `environment_versions.json`.

The scripts expect a workspace with this layout:

- `outputs/tiny_topology/`: protocols, programs, reports and final predictions.
- `outputs/real_topology/`: `optdigits_pilot.py` and `optdigits_ablation.py`, the frozen original-image loader and feature extractor.
- `outputs/topology_complement/`: upstream OptDigits feature-cache protocol.
- `work/real_data/optdigits.zip`: original UCI data archive.
- `work/real_data/optdigits_complement_trainval_v1.npz`: raw TRA/CV feature groups used by the MLP study.
- `work/tiny_topology/`: intermediate arrays, individual fits and final frozen weights.

The report package preserves these relative paths for the included files. No RIM-ONE raw images are redistributed in this package. OptDigits credit: E. Alpaydin and C. Kaynak, UCI Machine Learning Repository (1998), DOI 10.24432/C50P49, CC BY 4.0.

The original MLP entry point enumerated 24 impossible late-fusion configurations at the 256-weight budget and was interrupted after the feasible jobs mostly completed. The corrected recovery entry point `collect_tiny_mlp.py` preserves all existing completed jobs, excludes only algebraically infeasible entries and fills the missing feasible jobs. Use this entry point for a full feasible MLP reproduction, not the original unfiltered grid loop. The event is recorded in `tiny_mlp_execution_amendment.json` and `tiny_mlp_feasibility_audit.json`.

The completed experiment consists of:

1. `collect_tiny_mlp.py`: feasible TRA/CV MLP development (456 fits).
2. `tiny_cnn_pilot.py --workers 4`: TRA/CV compact CNN development (288 fits).
3. `tiny_audit.py --complete`: verify every development probability file and independent mathematical checks.
4. `tiny_confirmation.py freeze`: select configurations by the fixed rule; replay chosen development checkpoints; freeze code/data hashes.
5. `tiny_confirmation.py train --workers 4`: refit the 80 preselected single-model runs on TRA+CV, before test access.
6. `tiny_confirmation.py predict`: save all WDEP predictions before computing accuracy.
7. `tiny_confirmation.py stats`: compute the primary test and the descriptive budget curve.
8. `tiny_final_audit.py`: independent numerical and duplicate audits.
9. `tiny_export.py`: export seed0 primary MLP weights, folding fitted feature scaling into the existing affine weights. Seed0 is fixed, not selected by test accuracy.
10. `tiny_figures.py --final` and `tiny_report.py`: generate figures and the report.

Programs intentionally refuse to overwrite certain frozen files. Existing hashes certify this particular run, not an arbitrary changed experiment. Repeating or modifying the development study should be done in a separate copy, with new protocols and output files. WDEP has been evaluated once for this research question; it must not be called an untouched confirmation set for further tuning informed by these results.

A separately registered descriptive check, `tiny_seen_windep_replay.py`, evaluates the same final primary weights on the already-used older WINDЕP partition, only after the fresh WDEP analysis. This is not another untouched confirmation set; no models are chosen or refitted and no additional significance claim is made from that reuse.

Parameter budgets include every trainable weight and bias in a single model. They do not include morphology/component-labeling operations, code, temporary buffers or the memory of the training procedure. The exported numerical matrices can absorb input normalization, but the fixed feature extractor still costs computation and memory.
