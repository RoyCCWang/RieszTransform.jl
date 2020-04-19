# Module for first-order Riesz transform for signals that map ℝᴰ to ℝ,
#   given an uniformly sampled version of them.
# Implementation is done through the frequency domain via FFT.

module RieszTransform

    using LinearAlgebra
    using FFTW

    import Combinatorics
    import Interpolations

    import SymTensorTools
    import SignalTools

    # indirect dependencies.
    import Utilities

    include("wavelet.jl");
    include("riesz_filters.jl");
    include("monogenicanalysis.jl");
    include("riesz_wavelet.jl");


    export  getRTfilters, gethigherorderRTfilters, RieszAnalysis, RieszSynthesis,
            RieszAnalysisLimited, RieszSynthesisLimited,
            directionalHilbert, monogenicanalysis, instantfreq,
            getprefilters,
            waveletanalysis, waveletsynthesis,
            RieszwaveletAnalysis, convert𝓡ψtoψ𝓡,
            RieszwaveletSynthesis, convert𝓡ψtoψ𝓡vectorfield

end
