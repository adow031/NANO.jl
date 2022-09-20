module NANO

using Ipopt, JuMP

struct NeuralNetwork
    num_layers::Int
    num_nodes::Vector{Int}
    A::Vector{Array{Float64,2}}
    activation_func::Vector{Vector{Tuple{Symbol,Int}}}
end

function NeuralNetwork(
    num_inputs::Int,
    hidden_layers::Int,
    hidden_activations::Vector{Tuple{Symbol,Int}},
    target_activations::Vector{Tuple{Symbol,Int}},
)
    num_layers = hidden_layers + 2

    nodes_per_layer = 0
    for (func, count) in hidden_activations
        nodes_per_layer += count
    end

    num_outputs = 0
    for (func, count) in target_activations
        num_outputs += count
    end

    num_nodes = Int[]
    push!(num_nodes, num_inputs + 1)
    for i in 1:hidden_layers
        push!(num_nodes, nodes_per_layer + 1)
    end
    push!(num_nodes, num_outputs)

    A = Array{Float64,2}[]

    for i in 1:length(num_nodes)-2
        push!(
            A,
            rand(num_nodes[i], num_nodes[i+1] - 1) -
            0.5 * ones(num_nodes[i], num_nodes[i+1] - 1),
        )
    end
    push!(
        A,
        rand(num_nodes[end-1], num_nodes[end]) -
        0.5 * ones(num_nodes[end-1], num_nodes[end]),
    )

    af = Vector{Tuple{Symbol,Int}}[]

    for i in 1:num_layers-2
        push!(af, deepcopy(hidden_activations))
        push!(af[end], (:identity, 1))
    end
    push!(af, deepcopy(target_activations))

    return NeuralNetwork(num_layers, num_nodes, A, af)
end

function activation(x::Array{Float64,2}, groups::Vector{Tuple{Symbol,Int}})
    lines = 1
    out = zeros(size(x)[1])
    for (func, count) in groups
        z = x[:, lines:lines+count-1]

        if func == :sigmoid
            temp = 2 ./ (ones(size(z)) + exp.(-z)) - ones(size(z))
        elseif func == :positive
            temp = (z .> 0) .* z
        elseif func == :identity
            temp = z
        elseif func == :probability
            y = (z .> 0) .* z
            temp = y ./ sum(y[:, i] for i in 1:size(y)[2]) .* (y .> 0)
        elseif func == :elbow
            temp = (z .> 0) .* z + (z .< 0) .* z * 0.1
        elseif func == :sigmoid2
            temp = 1 ./ (ones(size(z)) + exp.(-z))
        elseif func == :quadratic
            temp = 2 ./ (ones(size(z)) + exp.(-(z .^ 2))) - ones(size(z))
        else
            error("Unknown activation function $func")
        end

        lines += count

        out = [out temp]
        if lines > size(x)[2]
            break
        end
    end

    return out[:, 2:end]
end

function derivative_activation(x::Array{Float64,2}, groups::Vector{Tuple{Symbol,Int}})
    lines = 1
    out = zeros(size(x)[1])
    for (func, count) in groups
        z = x[:, lines:lines+count-1]

        if func == :sigmoid
            temp = 2 * exp.(-z) ./ (ones(size(z)) + exp.(-z)) .^ 2
        elseif func == :positive
            temp = convert(Matrix{Float64}, z .> 0.0)
        elseif func == :identity
            temp = ones(size(z))
        elseif func == :probability
            y = (z .> 0) .* z
            temp =
                (
                    (ones(size(z)) .* sum(y[:, i] for i in 1:size(z)[2]) - y) ./
                    sum(y[:, i] for i in 1:size(z)[2]) .^ 2
                ) .* (z .> 0)
        elseif func == :elbow
            temp = (z .> 0.0) .* ones(size(z)) + (z .< 0) .* ones(size(z)) * 0.1
        elseif func == :sigmoid2
            temp = exp.(-z) ./ (ones(size(z)) + exp.(-z)) .^ 2
        elseif func == :quadratic
            temp = 4 * exp.(-(z .^ 2)) ./ (ones(size(z)) + exp.(-(z .^ 2))) .^ 2 .* z
        else
            error("Unknown activation function $func")
        end

        lines += count

        out = [out temp]
    end

    return out[:, 2:end]
end

function forward(input::Union{Vector{Float64},Array{Float64,2}}, nn::NeuralNetwork)
    temp = []
    act = []

    push!(temp, [input ones(size(input)[1])])
    push!(act, [input ones(size(input)[1])])

    for i in 1:nn.num_layers-2
        next = act[end] * nn.A[i]
        push!(temp, [next ones(size(next)[1])])
        push!(act, [activation(next, nn.activation_func[i]) ones(size(next)[1])])
    end
    next = act[end] * nn.A[end]
    push!(temp, next)
    push!(act, activation(next, nn.activation_func[end]))

    return temp, act
end

function backward(
    net::Vector{Any},
    active::Vector{Any},
    target::Union{Vector{Float64},Array{Float64,2}},
    nn::NeuralNetwork,
)
    dZ = active[end] - target

    del_j = Dict()

    del_j[nn.num_layers-1] = dZ .* derivative_activation(net[end], nn.activation_func[end])

    notfirst = 0
    for i in reverse(1:nn.num_layers-2)
        del_j[i] =
            del_j[i+1][:, 1:end-notfirst] * transpose(nn.A[i+1]) .*
            derivative_activation(net[i+1], nn.activation_func[i])
        notfirst = 1
    end
    return del_j
end

function update_weights(active::Vector{Any}, del_j::Dict, nn::NeuralNetwork, eta::Float64)
    for i in 1:nn.num_layers-1
        if i == nn.num_layers - 1
            extra = 0
        else
            extra = 1
        end
        delta = zeros(Float64, size(active[i])[2], size(del_j[i])[2] - extra)
        for j in 1:size(active[i])[1]
            delta += active[i][j, :] * transpose(del_j[i][j, 1:end-extra])
        end

        nn.A[i] -= delta / size(active)[1] * eta
    end
end

function train_nn(
    nn::NeuralNetwork,
    input::Union{Vector{Float64},Array{Float64,2}},
    target::Union{Vector{Float64},Array{Float64,2}},
    iterations::Int,
    eta::Float64,
)
    for i in 1:iterations
        net, active = forward(input, nn)
        del_j = backward(net, active, target, nn)
        update_weights(active, del_j, nn, eta)
    end
    return forward(input, nn)[2][end]
end

function loss(
    output::Union{Vector{Float64},Array{Float64,2}},
    target::Union{Vector{Float64},Array{Float64,2}},
)
    return sum((output - target) .^ 2)
end

function optimize_nn(
    nn::NeuralNetwork,
    input::Union{Vector{Float64},Array{Float64,2}},
    target::Union{Vector{Float64},Array{Float64,2}},
    regularization::Symbol,
    r::Real,
)
    if (regularization ∉ [:bound, :penalty])
        error("Unknown regularization method")
    end

    samplesize = nothing
    if typeof(input) == Vector{Float64}
        samplesize = length(input)
    else
        samplesize = size(input)[1]
    end

    nn.num_nodes[end] += 1

    model = JuMP.Model(Ipopt.Optimizer)

    if regularization == :bound
        @variable(
            model,
            -r <=
            w[i in 1:nn.num_layers-1, j in 1:nn.num_nodes[i], k in 1:nn.num_nodes[i+1]-1] <=
            r
        )
    else
        @variable(
            model,
            w[i in 1:nn.num_layers-1, j in 1:nn.num_nodes[i], k in 1:nn.num_nodes[i+1]-1]
        )
    end

    @variable(model, output[i in 1:nn.num_nodes[end]])
    @variable(
        model,
        node_value[i in 2:nn.num_layers, j in 1:nn.num_nodes[i], k in 1:samplesize]
    )
    @variable(
        model,
        node_value2[i in 1:nn.num_layers, j in 1:nn.num_nodes[i], k in 1:samplesize]
    )

    @constraint(
        model,
        fix_input[j in 1:nn.num_nodes[1]-1, k in 1:samplesize],
        node_value2[1, j, k] == input[k, j]
    )
    @constraint(
        model,
        fix_input2[j in [nn.num_nodes[1]], k in 1:samplesize],
        node_value2[1, j, k] == 1.0
    )

    for i in 1:nn.num_layers-1
        @constraint(
            model,
            [j in 1:nn.num_nodes[i+1]-1, k in 1:samplesize],
            node_value[i+1, j, k] ==
            sum(w[i, l, j] * node_value2[i, l, k] for l in 1:nn.num_nodes[i])
        )
        index = 1
        for (func, count) in nn.activation_func[i]
            if func == :sigmoid
                @NLconstraint(
                    model,
                    [j in index:index+count-1, k in 1:samplesize],
                    node_value2[i+1, j, k] ==
                    2.0 / (1.0 + exp(-node_value[i+1, j, k])) - 1.0
                )
            elseif func == :identity
                @constraint(
                    model,
                    [j in index:index+count-1, k in 1:samplesize],
                    node_value2[i+1, j, k] == node_value[i+1, j, k]
                )
            elseif func == :sigmoid2
                @NLconstraint(
                    model,
                    [j in index:index+count-1, k in 1:samplesize],
                    node_value2[i+1, j, k] == 1.0 / (1.0 + exp(-node_value[i+1, j, k]))
                )
            elseif func == :quadratic
                @NLconstraint(
                    model,
                    [j in index:index+count-1, k in 1:samplesize],
                    node_value2[i+1, j, k] ==
                    2.0 / (1.0 + exp(-node_value[i+1, j, k]^2)) - 1.0
                )
            end
            if index + count - 1 == nn.num_nodes[i+1] - 1
                break
            end
            index += count
        end
        @constraint(
            model,
            [k in 1:samplesize],
            node_value2[i+1, nn.num_nodes[i+1], k] == 1.0
        )
    end
    if regularization == :bound
        @NLobjective(
            model,
            Min,
            sum(
                (node_value2[nn.num_layers, j, k] - target[k, j])^2 for k in 1:samplesize,
                j in 1:nn.num_nodes[end]-1
            )
        )
    else
        @NLobjective(
            model,
            Min,
            sum(
                (node_value2[nn.num_layers, j, k] - target[k, j])^2 for k in 1:samplesize,
                j in 1:nn.num_nodes[end]-1
            ) +
            r * sum(
                w[i, j, k]^2 for i in 1:nn.num_layers-1, j in 1:nn.num_nodes[i],
                k in 1:nn.num_nodes[i+1]-1
            )
        )
    end

    for i in 1:nn.num_layers-1
        for j in 1:nn.num_nodes[i]
            for k in 1:nn.num_nodes[i+1]-1
                JuMP.set_start_value(w[i, j, k], nn.A[i][j, k])
            end
        end
    end

    optimize!(model)

    for i in 1:nn.num_layers-1
        for j in 1:nn.num_nodes[i]
            for k in 1:nn.num_nodes[i+1]-1
                nn.A[i][j, k] = value(w[i, j, k])
            end
        end
    end

    output = zeros(size(target))

    for k in 1:samplesize
        for j in 1:nn.num_nodes[end]-1
            output[k, j] = value(node_value2[nn.num_layers, j, k])
        end
    end

    nn.num_nodes[end] -= 1

    @info("Loss: $(loss(target,output))")
    return output
end
