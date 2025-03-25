using Test: @testset, @test, @inferred
using Statistics: mean
using RestrictedBoltzmannMachines: RBM, BinaryRBM, sample_v_from_v, sample_h_from_v
using StackedRBMs: pcd!

rbms = (
    BinaryRBM(randn(7), randn(5), randn(7,5)),
    BinaryRBM(randn(5), randn(3), randn(5,3))
)

data = rand(Bool, 5, 10)

pcd!(rbms[2], rbms[1], data)
