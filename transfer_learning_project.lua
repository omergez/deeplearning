require 'nn'
require 'image'
require 'optim'

logger = optim.Logger('Transfer.log') -- logger can be changed  
logger:setNames{'Trainset Error', 'Testset Error'}

local numClasses = 8

dataset = torch.load('flowers.t7')

dataset = dataset:narrow(1,1,numClasses)

classes = torch.range(1,numClasses):totable()
labels = torch.range(1,numClasses):view(numClasses,1):expand(numClasses,80)

print('dataset size:')
print(dataset:size()) --each class has 80 images of 3x128x128

image.display(dataset:select(2,3))

function shuffle(data,ydata) --shuffle data function
    local RandOrder = torch.randperm(data:size(1)):long()
    return data:index(1,RandOrder), ydata:index(1,RandOrder)
end



shuffledData, shuffledLabels = shuffle(dataset:view(-1,3,128,128), labels:contiguous():view(-1))

trainSize = 0.85 * shuffledData:size(1)
trainData, testData = unpack(shuffledData:split(trainSize, 1))
trainLabels, testLabels = unpack(shuffledLabels:split(trainSize, 1))

print('tain data size:')
print(trainData:size())

trainData = trainData:float() -- convert the data from a ByteTensor to a float Tensor.
trainLabels = trainLabels:float()

mean, std = trainData:mean(), trainData:std()

print('mwan, std:')
print(mean, std)

trainData:add(-mean):div(std)
    
testData = testData:float()
testLabels = testLabels:float()
testData:add(-mean):div(std)


-- Load GoogLeNet
googLeNet = torch.load('GoogLeNet_v2_nn.t7')

-- The new network
model = nn.Sequential()

for i=1,10 do
    local layer = googLeNet:get(i):clone()
    layer.parameters = function() return {} end --disable parameters
    layer.accGradParamters = nil --remove accGradParamters
    model:add(layer)
end

-- Check output dimensions with random input
model:float()
local y = model:forward(torch.rand(1,3,128,128):float())

print('Google net size:')
print(y:size())

-- Add the new layers

model:add(nn.SpatialConvolution(320, numClasses, 3, 3))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(4,4,4,4))
model:add(nn.View(16*3*3)) 
model:add(nn.Dropout())
model:add(nn.Linear(16*3*3, numClasses))  
model:add(nn.LogSoftMax())
model:float() 
print(tostring(model))

-- Ex3 start --


-- Loss Function = Negative Log Likelihood ()
lossFunc = nn.ClassNLLCriterion():float() 
w, dE_dw = model:getParameters()

print('Number of parameters:', w:nElement())

batchSize = 32
epochs = 200
optimState = {
    learningRate = 0.001,
    
}



function forwardNet(data, labels, train)
    --another helpful function of optim is ConfusionMatrix
    local confusion = optim.ConfusionMatrix(torch.range(1,numClasses):totable())
    local lossAcc = 0
    local numBatches = 0
    if train then
        --set network into training mode
        model:training()
    end
    
        
        numBatches = numBatches + 1
        local x = data:narrow(1, 1, batchSize):float()
        local yt = labels:narrow(1, 1, batchSize):float()
        local y = model:forward(x)
        local err = lossFunc:forward(y, yt)
        lossAcc = lossAcc + err
        confusion:batchAdd(y,yt)
        
        if train then
            function feval()
                model:zeroGradParameters() --zero grads
                local dE_dy = lossFunc:backward(y,yt)
                model:backward(x, dE_dy) -- backpropagation
            
                return err, dE_dw
            end
            
            --optim.adagrad(feval, w, optimState)
            optim.adam(feval, w, optimState)
            
 
        end
    
    
    confusion:updateValids()
    local avgLoss = lossAcc / numBatches
    local avgError = 1 - confusion.totalValid
    
    return avgLoss, avgError, tostring(confusion) 
end

trainLoss = torch.Tensor(epochs)
testLoss = torch.Tensor(epochs)
trainError = torch.Tensor(epochs)
testError = torch.Tensor(epochs)

function isEarlyStopping(testError)
	if (numClasses == 4 and testError <= 0.1) then return true end
	if (numClasses == 8 and testError <= 0.15) then return true end 
	if (numClasses == 12 and testError <= 0.2) then return true end
	if (numClasses == 16 and testError <= 0.2) then return true end

	return false
end

for e = 1, epochs do
    trainData, trainLabels = shuffle(trainData, trainLabels) --shuffle training data
    trainLoss[e], trainError[e] = forwardNet(trainData, trainLabels, true)
    testLoss[e], testError[e], confusion = forwardNet(testData, testLabels, false)
    logger:add{trainError[e],testError[e]} -- loss is the value which you want to plot
    logger:style{'-','-'}   -- the style of your line, as in MATLAB, we use '-' or '|' etc.
 
 		print('------------------------------------------')
 		print('------------------------------------------')
        print('Epoch ' .. e .. ':')
        print('Training error: ' .. trainError[e], 'Training Loss: ' .. trainLoss[e])
        print('Test error: ' .. testError[e], 'Test Loss: ' .. testLoss[e])
        print(confusion)
        
        if isEarlyStopping(testError[e]) then 
        	print("Early Stopping")
          	break
        end

        print('------------------------------------------')
        print('------------------------------------------')
    
end

   image.display(trainData:narrow(1,1,10))  

   print("Train classes label: ")
   print(trainLabels:narrow(1,1,10))

   local z = model:forward(torch.rand(1,3,128,128):float())
   
   print("LogSoftMax:")
   print(z)

   
logger:plot()
