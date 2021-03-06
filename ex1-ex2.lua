require 'nn'
require 'image'
require 'optim'

logger = optim.Logger('Transfer.log') -- logger can be changed  
logger:setNames{'Trainset Error', 'Testset Error'}

local numClasses = 16

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

model:add(nn.SpatialConvolution(320, 16, 3, 3))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(4,4,4,4))
model:add(nn.View(16*3*3)) 
model:add(nn.Dropout())
model:add(nn.Linear(16*3*3, numClasses))  
model:add(nn.LogSoftMax())
model:float() 
print(tostring(model))


