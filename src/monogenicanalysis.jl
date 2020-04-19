# The directional Hilbert transform and monogenic analysis functions require
#   a proper (i.e ℝ-valued ℛᵢf, ∀i∈[D]) Riesz transformed signal. This means
#   f should be isotropically bandlimited.

# compute Hᵤℛf, the 1-D Hilbert transform along the common direction u, for every entry
#   of the sampled signal f.
# assume f is isotropically bandlimited, so ℛf[i] are real-valued arrays.
function directionalHilbert( ℛf::Vector{Array{T,D}},
                                u::Vector{T})::Array{T,D} where {T,D}
    @assert !isempty(ℛf)
    @assert length(ℛf) == D

    Hᵤℛf = zeros(T, size(ℛf[1]))
    for d = 1:D
        for i = 1:length(ℛf[i])
            Hᵤℛf[i] += ℛf[d][i]*u[d]
        end
    end

    return Hᵤℛf # this is the Hilbert pair (wrt f, in dir. u)'s norm, squared.
end

# compute Hᵤℛf, the 1-D Hilbert transform along the direction specified in u_set,
#   for every entry of the sampled signal f. u_set specifies a different dir. for
#   each sampling location.
# assume f is isotropically bandlimited, so ℛf[i] are real-valued arrays.
function directionalHilbert( ℛf::Vector{Array{T,D}},
                             u_set::Array{Vector{T},D})::Array{T,D}  where {T,D}
    @assert !isempty(ℛf)
    @assert length(ℛf) == D

    Hᵤℛf = zeros(T, size(ℛf[1]))
    for d = 1:D
        @assert size(ℛf[d]) == size(u_set)

        for i = 1:length(ℛf[d])
            Hᵤℛf[i] += ℛf[d][i]*u_set[i][d]
        end
    end

    return Hᵤℛf # this is the Hilbert pair (wrt f, in dir. u)'s norm, squared.
end

# This version just returns the magnitude. Applicable to higher-order Riesz transforms.
# when we set the direction at each location to be
#   the Riesz transform vector at each sampling location.
function directionalHilbert(ℛf::Vector{Array{T,D}})::Array{T,D}  where {T,D}
    M = length(ℛf)
    @assert M > 0

    ℛf_norm_squared = zeros(T, size(ℛf[1]))
    for m = 1:M
        for i = 1:length(ℛf[m])
            ℛf_norm_squared[i] += ℛf[m][i]*ℛf[m][i]
        end
    end

    return ℛf_norm_squared # this is the Hilbert pair (wrt f, in dir. u)'s norm, squared.
end

# for higher-order Riesz transforms that has a diagonal metric tensor over the
#   M distinct channels.
# Returns just the magnitudes, but with the multinomial coefficient weighting.
function directionalHilbert(metric_tensor_vec::Vector,
                            ℛf::Vector{Array{T,D}})::Array{T,D}  where {T,D}
    @assert !isempty(ℛf)

    M = length(ℛf)
    @assert length(metric_tensor_vec) == M

    ℛf_norm_squared = zeros(T, size(ℛf[1]))
    for m = 1:M
        #weight_squared = Combinatorics.multinomial(α_array[m]...)

        for i = 1:length(ℛf[m])
            ℛf_norm_squared[i] += metric_tensor_vec[m]*ℛf[m][i]*ℛf[m][i]
        end
    end

    return ℛf_norm_squared # this is the Hilbert pair (wrt f, in dir. u)'s norm, squared.
end

# perform 1-D Hilbert transform along direction u, then do
#   analytic signal analysis. Do for every entry of f.
# assume f is isotropically bandlimited.
function monogenicanalysis( ℋf::Array{T,D},
                            f::Array{T,D})::Tuple{Array{T,D},Array{T,D}}  where {T,D}

    # instantaneous amplitude and phase along the dir. of u.
    Aᵤ = Array{T}(undef, size(f))
    ϕᵤ = Array{T}(undef, size(f))

    for i = 1:length(f)
        Aᵤ[i] = sqrt( ℋf[i]^2 + f[i]^2 )
        ϕᵤ[i] = atan(ℋf[i], f[i])
        #ϕᵤ[i] = atan(sqrt(ℛf_norm_squared[i]), f[i])
    end

    return Aᵤ, ϕᵤ
end

# for a common direction u.
function instantfreq( ϕᵤ::Array{T,D},
                      u::Vector{T},
                      A::Array{T,D})::Array{T,D} where {T,D}


    # compute gradient of instant phase along dir. u.
    itp = Interpolations.interpolate(A, Interpolations.BSpline(Interpolations.Cubic(Interpolations.Free())), Interpolations.OnGrid())

    ∇ϕᵤ = Array{Vector{T}}(undef, size(ϕᵤ))
    for linear_i = 1:length(ϕᵤ)
        i = ind2sub(size(ϕᵤ), linear_i)

        ∇ϕᵤ[i] = Interpolations.gradient(itp,i...)
    end

    # compute instant freq. as dir. derivative along u of instant phase.
    ζᵤ = Array{T}(undef, size(ϕᵤ))
    for i = 1:length(ℛf[i])
        ζᵤ[i] = dot(∇ϕᵤ[i],u)
    end

    return ζᵤ
end

# for the case of different directions at different sampling locations.
function instantfreq(   ϕᵤ::Array{T,D},
                        u_set::Array{Vector{T},D},
                        A::Array{T,D})::Tuple{Array{T,D},Array{Vector{T},D}} where {T,D}


    # compute gradient of instant phase along dir. u.
    itp = Interpolations.interpolate(A, Interpolations.BSpline(Interpolations.Cubic(Interpolations.Free())), Interpolations.OnGrid())

    ∇ϕᵤ = Array{Vector{T}}(undef, size(ϕᵤ))
    for linear_i = 1:length(ϕᵤ)
        i = ind2sub(size(ϕᵤ), linear_i)

        ∇ϕᵤ[linear_i] = Interpolations.gradient(itp,i...)
    end

    # compute instant freq. as dir. derivative along u of instant phase.
    ζᵤ = Array{T}(undef, size(ϕᵤ))
    for i = 1:length(ϕᵤ)
        ζᵤ[i] = dot(∇ϕᵤ[i],u_set[i])
    end

    return ζᵤ, ∇ϕᵤ
end

# for the case of using the Riesz transform vector direction at each sampling locations.
function instantfreq(   ϕᵤ::Array{T,D},
                        ℛf::Vector{Array{T,D}},
                        A::Array{T,D})::Tuple{Array{T,D},Array{Vector{T},D}}  where {T,D}


    # compute gradient of instant phase along dir. u.
    itp = Interpolations.interpolate(A,
            Interpolations.BSpline(Interpolations.Cubic(Interpolations.Free(Interpolations.OnGrid()))))

    LUT_sz = CartesianIndices(size(ϕᵤ))

    ∇ϕᵤ = Array{Vector{T}}(undef, size(ϕᵤ))
    for linear_i = 1:length(ϕᵤ)
        i = LUT_sz[linear_i]

        ∇ϕᵤ[linear_i] = Interpolations.gradient(itp, Tuple(i)...)
    end

    # compute instant freq. as dir. derivative along u of instant phase.
    ζᵤ = zeros(T,size(ϕᵤ))
    for i = 1:length(ϕᵤ)
        for d = 1:D
            ζᵤ[i] = ∇ϕᵤ[i][d]*ℛf[d][i]
        end
    end

    return ζᵤ, ∇ϕᵤ
end
