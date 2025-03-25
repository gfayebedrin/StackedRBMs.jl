function pcd!(
    rbm::RBM,
    rbm_below::Union{RBM,StandardizedRBM},
    data_repr::AbstractArray; # visible layer of the RBM being trained
    batchsize::Int=1,
    iters::Int=1, # number of gradient updates
    wts::Union{AbstractVector,Nothing}=nothing, # data weights
    steps::Int=1, # MC steps to update fantasy chains
    optim::AbstractRule=Adam(), # optimizer rule
    moments=moments_from_samples(rbm.visible, data_repr; wts), # sufficient statistics for visible layer

    # Kullback-Leibler
    α_KL::Real=0, # KL divergence weight

    # regularization
    l2_fields::Real=0, # visible fields L2 regularization
    l1_weights::Real=0, # weights L1 regularization
    l2_weights::Real=0, # weights L2 regularization
    l2l1_weights::Real=0, # weights L2/L1 regularization

    # gauge
    zerosum::Bool=true, # zerosum gauge for Potts layers
    rescale::Bool=true, # normalize weights to unit norm (for continuous hidden units only)
    callback=Returns(nothing), # called for every batch

    # init fantasy chains
    vm=sample_from_inputs(rbm.visible, Falses(size(rbm.visible)..., batchsize)),
    hm_below=sample_from_inputs(rbm_below.hidden, Falses(size(rbm_below.hidden)..., batchsize)), shuffle::Bool=true,

    # parameters to optimize
    ps=(; visible=rbm.visible.par, hidden=rbm.hidden.par, w=rbm.w),
    state=setup(optim, ps) # initialize optimiser state
)

    @assert size(data_repr) == (size(rbm.visible)..., size(data_repr)[end])
    @assert isnothing(wts) || size(data_repr)[end] == length(wts)

    # gauge constraints
    zerosum && zerosum!(rbm)
    rescale && rescale_weights!(rbm)

    # store average weight of each data point
    wts_mean = isnothing(wts) ? 1 : mean(wts)

    for (iter, (vd, wd)) in zip(1:iters, infinite_minibatches(data_repr, wts; batchsize, shuffle))
        # update fantasy chains
        vm .= sample_v_from_v(rbm, vm; steps)
        hm_below .= sample_h_from_h(rbm_below, hm_below; steps)

        # compute gradient
        ∂d = ∂free_energy(rbm, vd; wts=wd, moments)
        ∂m = ∂free_energy(rbm, vm)
        ∂m_below = ∂free_energy(rbm, hm_below)
        ∂ = ∂d + α_KL * ∂m_below - (1 + α_KL) * ∂m

        # correct weighted minibatch bias
        batch_weight = isnothing(wts) ? 1 : mean(wd) / wts_mean
        ∂ *= batch_weight

        # weight decay
        ∂regularize!(∂, rbm; l2_fields, l1_weights, l2_weights, l2l1_weights, zerosum)

        # feed gradient to Optimiser rule
        gs = (; visible=∂.visible, hidden=∂.hidden, w=∂.w)
        state, ps = update!(state, ps, gs)

        # reset gauge
        rescale && rescale_weights!(rbm)
        zerosum && zerosum!(rbm)

        callback(; rbm, optim, state, iter, vm, vd, wd)
    end
    return state, ps

end
