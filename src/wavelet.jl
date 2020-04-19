# Only Simoncelli lowpass/highpass pairs are used.
# The wavelet analysis assumes an input signal f: ℝᴰ → ℝ.
# Since Simoncelli filters have only real components, and we're doing
#   linear filtering, the wavelet coefficients are also real-valued.


function getprefilters(Y::Array{T,D}) where {T,D}
    return SignalTools.getSimoncellifilters(size(Y), Val(D), 2.0)
end

# function applyprefilters(y)
#     LP,HP = RieszTransform.getprefilters(y)
#     Y = real.(ifft(fft(y).*LP)) # bandlimited version of y.
#     residual = real.(ifft(fft(y).*HP))
#
#     return Y, residual
# end


# isotropic band reductions across scales.
# uses flters that are strictly real and symmetric.
function waveletanalysis(   Y::Array{T,D},
                            N_scales::Int)::Vector{Array{T,D}} where {T,D}

    Ŷ = fft(Y)
    Ŷ_LP = copy(Ŷ)
    ψY = Vector{Array{T,D}}(undef, N_scales)

    for s = 1:(N_scales-1)
        LP, HP = SignalTools.getSimoncellifilters(size(Y), Val(D), 1.0/2^(s-1))

        ψY[s] = real.(ifft(Ŷ_LP.*HP))
        Ŷ_LP = Ŷ_LP.*LP
    end
    ψY[end] = real.(ifft(Ŷ_LP))

    return ψY
end


function waveletsynthesis(ψY::Vector{Array{T,D}})::Array{T,D} where {T,D}

    N_scales = length(ψY)
    ψŶ = fft(ψY[end]) # this is fft(ψY[N_scales])

    for s = (N_scales-1):-1:1
        LP, HP = SignalTools.getSimoncellifilters(size(ψY[s]), Val(D), 1.0/2^(s-1))

        ψŶ = ψŶ.*LP + fft(ψY[s]).*HP
    end
    Y = real.(ifft(ψŶ))

    return Y
end
