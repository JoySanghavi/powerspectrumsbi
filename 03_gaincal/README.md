Files/folders to use are inside 03_gaincal. Specifically

1 - data folder - it has the observation. normed_tracked_uv.npy for the visibilities and normed_tracked_uvobs.npy for the observation image. Download both of them.

2 - src - the functionlistnew and model files are needed. You also need paddedAA2_4_10_9_128.pth for the antennae positions.

3 - config file for falcon configuration. z, pw are my inferred parameters (and here z can be commented out). obsx is my observed parameter - normed_tracked_uvobs.npy. If one prefers to use x (visibilities) as the observed parameter, the normed_tracked_uv.npy is to be used.

The other files are not completely necessary.