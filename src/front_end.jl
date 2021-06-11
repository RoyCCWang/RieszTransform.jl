function performmonogenicwaveletanalysis(  y::Array{T,D},
                                        N_scales = round(Int, log2( maximum(size(y)) )),
                                        scale_select = cld(N_scales,2) ) where {T,D}

    # get Riesz-wavelet transform.
    𝓡ψY,residual = RieszwaveletAnalysis(y, N_scales)

    # get wavelet bands. Need this for the real part of the monogenic signal.
    LP,HP = getprefilters(y)
    Y = real.(ifft(fft(y).*LP)) # bandlimited version of y.
    residual = real.(ifft(fft(y).*HP))
    ψY = waveletanalysis(Y, N_scales)

    # convert data structure.
    ψ𝓡Y = convert𝓡ψtoψ𝓡(𝓡ψY)

    Hᵤ𝓡ψY_ds = directionalHilbert(ψ𝓡Y[scale_select])
    Aᵤ, ϕᵤ = monogenicanalysis(Hᵤ𝓡ψY_ds, ψY[scale_select])
    ζᵤ, ∇ϕᵤ = instantfreq(ϕᵤ, ψ𝓡Y[scale_select], ψY[scale_select])

    ∇ϕᵤ_norm = collect( norm(∇ϕᵤ[i]) for i = 1:length(∇ϕᵤ) )

    return 𝓡ψY, ψY, Hᵤ𝓡ψY_ds, Aᵤ, ϕᵤ, ζᵤ, ∇ϕᵤ, ∇ϕᵤ_norm
end

function Rieszwaveletreconstructiondemo(y::Array{T,D},
                                    N_scales = round(Int, log2( maximum(size(y)) ))) where {T,D}

    println("Demo for Riesz-wavelet reconstruction.")
    𝓡ψY,residual = RieszwaveletAnalysis(y,N_scales)
    yr = RieszwaveletSynthesis(𝓡ψY,residual)
    println("discrepancy between y and yr: ", sum(abs.(y-yr)) )
    println()

end

function Rieszreconstructiondemo(A::Array{T,D}) where {T,D}

    LP,HP = getprefilters(A)
    Y = real.(ifft(fft(A).*LP))
    residual = real.(ifft(fft(A).*HP))
    println("Demo for Riesz transform reconstruction.")
    println("Y is isotropically bandlimited version of A.")

    H = getRTfilters(Y)
    B = RieszAnalysisLimited(Y,H)
    Yr = RieszSynthesisLimited(B,H)
    println("Discard imaginary parts: discrepancy between Y and Yr: ", sum(abs.(Y-Yr)) )

    H = getRTfilters(A)
    B = RieszAnalysisLimited(A,H)
    Ar = RieszSynthesisLimited(B,H)
    println("Discard imaginary parts: discrepancy between A and Ar: ", sum(abs.(A-Ar)), ". This should not be zero for a non-bandlimited A." )

    H = getRTfilters(A)
    B = RieszAnalysis(A,H)
    Ar = RieszSynthesis(B,H)
    println("Discard nothing: discrepancy between A and Ar: ", sum(abs.(A-Ar)) )
    println()

end

# for one frequency band.
function filterpairreconstructiondemo(A::Array{T,D}, N_tests = 100) where {T,D}
    (h,w,d) = size(A)

    total_discrepancy = 0.0
    for n = 1:N_tests
        s=rand()
        LP, HP = SignalTools.getSimoncellifilters(size(A), Val(D), 1.0/2^(s-1))
        Y = real.(ifft(fft(A).*LP))
        residual = real.(ifft(fft(A).*HP))
        total_discrepancy += sum(abs.(ifft(fft(Y).*LP+fft(residual).*HP)-A))
    end
    println("Demo for Lowpass and highpass reconstruction.")
    Printf.@printf("Total reconstruction discrepancies of %d trials: ", N_tests)
    println(total_discrepancy)
    println()
end

# for multiple frequency bands, which together makes a redundant wavelet analysis.
function redundantwaveletreconstructiondemo(A::Array{T,D}) where {T,D}
    (h,w,d) = size(A)

    LP,HP = getprefilters(A)
    Y = real.(ifft(fft(A).*LP))
    residual = real.(ifft(fft(A).*HP))

    levels = log2( maximum(size(Y)) )
    N_scales = round(Int, levels)
    ψY = waveletanalysis(Y, N_scales)
    Yr = waveletsynthesis(ψY)

    Ar = real.(ifft(fft(Yr).*LP + fft(residual).*HP))
    discrepancy = sum(abs.(A-Ar))
    println("Demo for Redundant wavelet reconstruction.")
    println("discrepancy: ", discrepancy)
    println()

    return discrepancy
end
