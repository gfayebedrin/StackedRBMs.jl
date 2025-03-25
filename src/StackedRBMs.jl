module StackedRBMs

using Base: front
using Base: tail
using ConditionalSwaps: swap
using MetropolisRules: metropolis_rule
using RestrictedBoltzmannMachines: free_energy_h
using RestrictedBoltzmannMachines: free_energy_v
using RestrictedBoltzmannMachines: sample_h_from_v
using RestrictedBoltzmannMachines: sample_v_from_h
using RestrictedBoltzmannMachines: sample_v_from_v
using RestrictedBoltzmannMachines: sample_h_from_h
using RestrictedBoltzmannMachines: RBM
using RestrictedBoltzmannMachines: StandardizedRBM
using Optimisers: AbstractRule, setup, update!, Adam
using Tail2Front2: tail2

include("stacked_training.jl")
include("stacked_tempering.jl")
include("swap.jl")

end
