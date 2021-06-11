
using FFTW

using LinearAlgebra
import Printf

import SymTensorTools
import Combinatorics
import Interpolations

import SignalTools

import Random

using Test

include("../src/monogenicanalysis.jl")
include("../src/riesz_filters.jl")
include("../src/riesz_wavelet.jl")
include("../src/wavelet.jl")

include("../src/front_end.jl")

Random.seed!(25)


A = rand(10,10,10)
D = length(size(A))

# This demo doesn't visualize the results, because I don't want to load PyPlot.
analysis_results = performmonogenicwaveletanalysis(A)

# Demos with verbose screen output for verification.
Rieszwaveletreconstructiondemo(A)
filterpairreconstructiondemo(A)
Rieszreconstructiondemo(A)


@test redundantwaveletreconstructiondemo(A) < 1e-12
#function visualizefilters()
    ## visualize filters
    # import PyPlot
    # fig_num = 1
    #
    # close("all")
    #
    # PyPlot.figure(fig_num)
    # fig_num += 1
    # PyPlot.imshow(abs2(LP[:,:,1]), interpolation="nearest", cmap="Greys_r")
    # PyPlot.plt[:colorbar]()
    # plot_title_string = "LP"
    # PyPlot.title(plot_title_string)
    #
    # PyPlot.figure(fig_num)
    # fig_num += 1
    # PyPlot.imshow(abs2(HP[:,:,1]), interpolation="nearest", cmap="Greys_r")
    # PyPlot.plt[:colorbar]()
    # plot_title_string = "HP"
    # PyPlot.title(plot_title_string)
#end
