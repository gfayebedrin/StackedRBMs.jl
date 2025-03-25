import Aqua
import StackedRBMs
using Test: @testset

@testset verbose = true "aqua" begin
    Aqua.test_all(StackedRBMs; ambiguities = false)
end
