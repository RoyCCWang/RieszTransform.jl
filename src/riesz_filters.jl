# The Riesz analysis and synthesis assumes the signal under analysis is
#   f: ℝᴰ → ℝ. Their bandlimited versions assume f is isotropically bandlimited.
#
# If f isn't isotropically bandlimited, then the implemented ℛᵢf is ℂ-valued,
#   for any channel i. This does not implement the Riesz transform, since its
#   output should be ℝ-valued (a isotropically low-pass, then derivative).
# If f is isotropically bandlimited, then ℛᵢf is ℝ-valued, like it should be.

function getunnormalizedRTfilters(Y::Array{T,D})::Vector{Array{Complex{T},D}} where {T,D}

    ω_grid = SignalTools.computefreqgrid(size(Y), Val(D))

    ## Riesz frequency rsp.
    # allocate and initialize stoage.
    H = collect( zeros(Complex{T},size(Y)) for d = 1:D )

    # at non-DC frequencies. Overwrite DC later.
    for d = 1:D

        #for i_sub in CartesianRange(size(Y))
        for i in CartesianIndices(size(Y))

            # fetch the corresponding multi-linear index
            #i = i_sub.I


            #if !mapreduce( x -> (x==1 ? true : false), &, i ) # skip DC.

            # get norm.
            ω_norm = zero(T)
            for dd = 1:D
                ω_norm += ω_grid[dd][i[dd]]^2
            end
            ω_norm = sqrt(ω_norm)

            # store filter to array.
            for dd = 1:D
                #H[dd][i...] = Complex( zero(T), -ω_grid[dd][i[dd]]/ω_norm )
                H[dd][i] = Complex( zero(T), -ω_grid[dd][i[dd]]/ω_norm )
            end
            #end

        end
    end

    # at DC frequency.
    for d = 1:D
        H[d][1] = Complex( one(T), zero(T))
    end

    # higher-order Riesz transform. Not implemented for now.
    # for n1=0:config.order
    #
    #     n2 = config.order-n1;
    #     n=[n1;n2];
    #     RieszFilters[iter]  = sqrt(multinomial(n))*base1.^(n1).*base2.^(n2);
    #     iter +=1;
    # end

    return H
end

function gethigherorderRTfilters(   Y::Array{T,D},
                                    order::Int)::Tuple{Vector{Array{Complex{T},D}},Vector{Vector{Int}}} where {T,D}

    ### get the first-order Riesz transform, without the DC normalization applied.
    H = getunnormalizedRTfilters(Y)

    M = binomial(order+D-1,D-1)

    # higher-order Riesz transform. Not implemented for now.
    # for n1=0:config.order
    #
    #     n2 = config.order-n1;
    #     n=[n1;n2];
    #     RieszFilters[iter]  = sqrt(multinomial(n))*base1.^(n1).*base2.^(n2);
    #     iter +=1;
    # end

    H_Riesz = collect( zeros(Complex{T}, size(Y)) for m = 1:M )

    # to do: use some monomial ordering, like from the symmetric tensor package.
    #   That way, we can handle arbitrary dimensions.
    α_array = SymTensorTools.getsubscriptarray(order, D)
    @assert M == length(α_array)

    m = 0
    for α in α_array
        m += 1
        tmp = sqrt(Combinatorics.multinomial(α...))
        fill!(H_Riesz[m],tmp)

        # do H.^α
        for d = 1:D
            if α[d] > 0
                H_Riesz[m] = H_Riesz[m].*(H[d].^α[d])
            elseif α[d] == 1
                H_Riesz[m] = H_Riesz[m].*H[d]
            end
        end
    end

    ### Apply DC normalization.
    # from Nicolas Chounard:
    # The 0 frequency is not well defined for the Riesz transform ( ||w|| at the denominator of the frequency response). It turns out not to be a problem when applying it to wavelet bands because they cancel the DC terms anyway. You could actually put any number there in the Riesz filters, it wouldn't change a thing.
    #
    # The DC term here can however have some importance in a very particular case: when you decide to switch the order in which you apply the Riesz and wavelet transforms: Riesz first, wavelet then. This could be useful in some case if you want to compute the Riesz transform only once for all scales at a time. This is what is done in the implementation I provide.
    # In that case you need to set up the DC term in the filters to some appropriate value so as to keep the self-invertibility property.
    #
    # Basically, I chose to normalize by dN = d^(0.5*config.RieszOrder); just as a scaling factor over all bands so that when you apply the Riesz filters, and then their adjoint (take the complex conjugate of the filters in Fourier domain) and sum the bands, then you get 1. It's not to hard to compute this factor from the formulas of the filters applied just before.This condition is necessary for the perfect reconstruction property of the filterbank. Basically you should have sum of abs^2 of the DC component of the Riesz filters to be 1. In fact, this should also be true at any other frequency in order to the Riesz filters to define a tight frame ie. self invertible:
    # conj(RW) x (RW) f  = f
    # This is how the multinomial coefficients for the filters have been decided in the first place.
    dN = D^(0.5*order);
    for m = 1:M
        H_Riesz[m][1] = H_Riesz[m][1]/dN
    end

    return H_Riesz, α_array
end

# for order 1. Discard the monomial orderings, since this case means the channels
#   correspond to the dimensions.
function getRTfilters(Y::Array{T,D})::Vector{Array{Complex{T},D}} where {T,D}
    return gethigherorderRTfilters(Y,1)[1]
end

## naive implementation of multinomial for 2 dimensions
##   no precomputing of n!
##       could do that later for speed up
# function multinomial(n::Array{Int,1})
#     @assert length(n) == 2 # only work on 2D signals
#     @assert !any(n.<0) # n must be non-negative integers
#     return multilinearfactorial(n)/n[1]/n[2];
# end
#
# function multilinearfactorial(multi_ind::Array{Int,1})
#     out::Int=1;
#     for i=1:length(multi_ind)
#         out=out*factorial(multi_ind[i]);
#     end
#     return out;
# end

# assumes f is not isotropically bandlimited.
function RieszAnalysis(f::Array{T,D},
                H::Vector{Array{Complex{T},D}})::Vector{Array{Complex{T},D}} where {T,D}

    # Rf = Array(Array{Complex{Float64},D}, D)
    # for d = 1:D
    #     Rf[d] = ifft(fft(f).*H[d])
    # end

    Rf = collect( ifft(fft(f).*H[m]) for m = 1:length(H) )

    return Rf
end

function RieszSynthesis(Rf::Vector{Array{Complex{T},D}},
                H::Vector{Array{Complex{T},D}})::Array{Complex{T},D} where {T,D}

    f = zeros(Complex{Float64}, size(Rf[1]))

    M = length(H)
    @assert M == length(Rf)

    for m = 1:M
        f += ifft(fft(Rf[m]).*conj(H[m]))
        #f += ifft(fft(real.(Rf[m])).*conj(H[m])) # you must make sure Rf was obtained from an isotropically bandlimited signal.
    end

    return f
end

# This is the proper Riesz transform.
# assumes f is isotropically bandlimited.
function RieszAnalysisLimited(f::Array{T,D},
                H::Vector{Array{Complex{T},D}})::Vector{Array{T,D}} where {T,D}

    fft_f = fft(f)
    Rf = collect( real.(ifft(fft_f.*H[m])) for m = 1:length(H) )

    return Rf
end

function RieszAnalysisLimited(fft_f::Array{Complex{T},D},
                H::Vector{Array{Complex{T},D}})::Vector{Array{T,D}} where {T,D}

    Rf = collect( real.(ifft(fft_f.*H[m])) for m = 1:length(H) )

    return Rf
end


# This is the proper inverse Riesz transform.
# assumes f is isotropically bandlimited.
function RieszSynthesisLimited(Rf::Vector{Array{T,D}},
                H::Vector{Array{Complex{T},D}})::Array{T,D} where {T,D}

    f = zeros(T, size(Rf[1]))

    M = length(H)
    @assert M == length(Rf)

    for m = 1:M
        f += real.(ifft(fft(real.(Rf[m])).*conj(H[m])))
    end

    return f
end
