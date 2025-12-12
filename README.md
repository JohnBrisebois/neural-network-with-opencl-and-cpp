- Requires NVIDIA GPU computing toolkit.  Modify path in run.bat to point to where you install it.
- Requires g++
- Requires openCL

To run, simply use run.bat

Parameters inside nnGPU.cpp:
- Topology is passed to the constructor { input, hidden... , output}.  Make sure input and output matches the length of your data
- Epoch count is NNGPU class constant "EPOCHS"
- Batch size is NNGPU class constant "BATCH_SIZE"
- input/output vectors must be the following format and be the same length:  { {float, float... }, { float, float ...} , ...    }
- predict() expects and returns a vector of floats.
    
