using NANO, DataFrames, CSV

cd(@__DIR__)

iris=CSV.read("iris.csv",DataFrame)
input = [iris[!,2] iris[!,3] iris[!,4] iris[!,5]]

function print_confusion(output,target)
    confusion=zeros(3,3)
    for i in 1:size(output)[1]
        o=argmax(output[i,:])
        t=argmax(target[i,:])
        confusion[t,o]+=1
    end
    println(confusion[1,:])
    println(confusion[2,:])
    println(confusion[3,:])
end

function predict_iris(input::Matrix,target::Matrix,hidden_layers::Int,nodes_per_layer::Int,method::Symbol)
    nn=NANO.NeuralNetwork(4,hidden_layers,[(:sigmoid,nodes_per_layer)],[(:sigmoid,3)])

    if method==:optimize
        output = NANO.optimize_nn(nn,input,target,:penalty,0.0001)
    elseif method==:backprop
        output = NANO.train_nn(nn,input,target,20000,0.05)
    else
        error("method must be either :optimize or :backprop")
    end

    loss = NANO.loss(output,target)

    for i in 1:size(output)[1]
        println("$(output[i,1]),$(output[i,2]),$(output[i,3])")
    end

    print_confusion(output,target)
    return nn, loss
end

# n = 150 training / in-sample predictions
target = zeros(150,3)
target[1:50,1].=1.0
target[51:100,2].=1.0
target[101:150,3].=1.0

predict_iris(input,target,1,1,:optimize)
predict_iris(input,target,1,2,:optimize)
predict_iris(input,target,2,1,:optimize)
predict_iris(input,target,2,2,:optimize)

predict_iris(input,target,1,1,:backprop)
predict_iris(input,target,1,2,:backprop)
predict_iris(input,target,2,1,:backprop)
predict_iris(input,target,2,2,:backprop)

# n = 75 training / out-of-sample predictions
input2 = zeros(75,4)
input2[1:25,:] = input[1:25,:]
input2[26:50,:] = input[51:75,:]
input2[51:75,:] = input[101:125,:]

target2 = zeros(75,3)
target2[1:25,1].=1.0
target2[26:50,2].=1.0
target2[51:75,3].=1.0

nn,loss = predict_iris(input2,target2,1,2,:backprop)

# prediction
input3 = zeros(75,4)
input3[1:25,:] = input[26:50,:]
input3[26:50,:] = input[76:100,:]
input3[51:75,:] = input[126:150,:]
output = NANO.forward(input3, nn)[2][end]
print_confusion(output,target2)
