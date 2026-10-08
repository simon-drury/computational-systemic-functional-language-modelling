# CSFLM-C001-ENCODERS

Three ways a model can read a 3x3 state, each returning one vector per state (one sequence position per state). Code: `sfl_pi_encoders.py` in this folder.

1. Flat control: nine cells, one linear map (the current SFL-pi behaviour).
2. Axis mixing: C channels compute A M B^T, A mixing metafunction rows and B mixing register columns, then GELU, then a linear map. The nonlinearity keeps it from collapsing into the flat control.
3. Diagonal core: diagonal cells form the main stream; the six off-diagonal cells form a context stream scaled by a gate computed from the diagonal. `use_context=False` is the ablation.

Status: written, never run in PyTorch. Not yet wired into any trainer. Parameter counts at d_model=128: flat 1,280; axis (8 channels) 9,488; diagonal core 1,920. Capacity should be matched before any comparison.
