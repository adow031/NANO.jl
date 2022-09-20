using NANO, DataFrames, CSV

cd(@__DIR__)

iris=CSV.read("iris.csv",DataFrame)
input = [iris[!,2] iris[!,3] iris[!,4] iris[!,5]]
input

target = zeros(150,3)
target[1:50,1].=1.0
target[51:100,2].=1.0
target[101:150,3].=1.0
target

function predict_iris(hidden_layers::Int,nodes_per_layer::Int)
    nn=NANO.NeuralNetwork(4,hidden_layers,[(:sigmoid,nodes_per_layer)],[(:sigmoid,3)])

    output = NANO.optimize_nn(nn,input,target,:penalty,0.0001)

    #output = NANO.train_nn(nn,input,target,10000,0.05)

    loss = NANO.loss(output,target)

    for i in 1:150
        println("$(output[i,1]),$(output[i,2]),$(output[i,3])")
    end

    count=0
    for i in 1:150
        o=argmax(output[i,:])
        t=argmax(target[i,:])

        if o==t
            mark = "✔"
            count+=1
        else
            mark = "✖"
        end

        println("$t, $o $mark")
    end

    println("$count/150 correct")
    return loss, count
end

predict_iris(1,1)
predict_iris(1,2)
predict_iris(2,1)
predict_iris(2,2)
