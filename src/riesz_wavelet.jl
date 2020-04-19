# functions for combining Riesz transform with isotropic, redundant wavelet
#   transform. Current implementation does Riesz transform first,
#   then redundant wavelet.
#
# Current implementation recreates the wavelet and Riesz filters as needed,
#   as opposed to storing them.

"""
RieszwaveletAnalysis(y::Array{T,D},
                              N_scales::Int)::Tuple{Vector{Vector{Array{T,D}}},Array{T,D}}
                              
returns 𝓡ψY, residual.
𝓡ψY[d][s][n], where:
- d is dimension index, up to D.
- s is scale index, up to N_scales.
- n is sampling position index, for a D-dim array.
"""
function RieszwaveletAnalysis(y::Array{T,D},
                              N_scales::Int)::Tuple{Vector{Vector{Array{T,D}}},Array{T,D}} where {T,D}
    LP,HP = getprefilters(y)
    Y = real.(ifft(fft(y).*LP)) # bandlimited version of y.
    residual = real.(ifft(fft(y).*HP))

    H = getRTfilters(Y)
    𝓡Y = RieszAnalysisLimited(Y,H)

    𝓡ψY = collect( waveletanalysis(𝓡Y[d], N_scales) for d = 1:D )

    return 𝓡ψY,residual
end

# convert data structures
function convert𝓡ψtoψ𝓡(𝓡ψY::Vector{Vector{Array{T,D}}})::Vector{Vector{Array{T,D}}} where {T,D}
    @assert !isempty(𝓡ψY)

    N_scales = length(𝓡ψY[1])
    ψ𝓡Y = Vector{Vector{Array{T,D}}}(undef, N_scales)

    for s = 1:N_scales
        ψ𝓡Y[s] = Vector{Array{T,D}}(undef, D)

        for d = 1:D
            ψ𝓡Y[s][d] = 𝓡ψY[d][s]
        end
    end

    return ψ𝓡Y
end

# convert data structures
function convert𝓡ψtoψ𝓡vectorfield(𝓡ψY::Vector{Vector{Array{T,D}}})::Vector{Array{Vector{T},D}} where {T,D}
    @assert !isempty(𝓡ψY)

    N_scales = length(𝓡ψY[1])
    ψ𝓡Y = Array{Array{Vector{T},D}}(undef, N_scales)

    for s = 1:N_scales
        ψ𝓡Y[s] = Array{Vector{T}}(undef, size(𝓡ψY[1][s]))

        for i = 1:length(𝓡ψY[1][s])
            ψ𝓡Y[s][i] = Vector{T}(undef, D)

            for d = 1:D
                ψ𝓡Y[s][i][d] = 𝓡ψY[d][s][i]
            end
        end
    end

    return ψ𝓡Y
end

function RieszwaveletSynthesis( 𝓡ψY::Vector{Vector{Array{T,D}}},
                                residual::Array{T,D})::Array{T,D} where {T,D}

    𝓡Yr = collect( waveletsynthesis(𝓡ψY[d]) for d = 1:D )

    @assert !isempty(𝓡ψY)
    H = getRTfilters(𝓡Yr[1])
    Yr = RieszSynthesisLimited(𝓡Yr,H)

    LP,HP = getprefilters(Yr)
    Ar = real.(ifft(fft(Yr).*LP + fft(residual).*HP))
end
