%% Artificial Neural Networks         %%
%% Kakkos Ioannis - 9304136030        %%
%% Classification challenge           %%
% clear all;

%% Data loading
[xTrain,tTrain,xValid,tValid,xTest,tTest] = LoadMNIST(3);
xTest2 = loadmnist2();
xTest = xTest2;

%% Network
layers = [
    imageInputLayer([28 28 1])
    convolution2dLayer(3,20,'Padding',1,'WeightsInitializer',...
    'narrow-normal','BiasInitializer','narrow-normal')
    batchNormalizationLayer
    reluLayer
    averagePooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,30,'Padding',1,'WeightsInitializer',...
    'narrow-normal','BiasInitializer','narrow-normal')
    batchNormalizationLayer
    reluLayer
    averagePooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,50,'Padding',1,'WeightsInitializer',...
    'narrow-normal','BiasInitializer','narrow-normal')
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(10,'WeightsInitializer','narrow-normal',...
    'BiasInitializer','narrow-normal')
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm',...
    'Momentum',0.9,...
    'MiniBatchSize',8192,...
    'ValidationData',{xValid,tValid},...
    'ValidationFrequency',30,...
    'MaxEpochs',30,...
    'ValidationPatience',5,...
    'Plots','training-progress',...
    'InitialLearnRate',0.01,...
    'L2Regularization',0,...
    'Shuffle','every-epoch');

[net,tr] = trainNetwork(xTrain,tTrain,layers,options);

predtTrain = net.classify(xTrain);
TrainAccuracy = sum(predtTrain == tTrain)/numel(tTrain);
TrainError = (1 - TrainAccuracy)^100;

predtValid = net.classify(xValid);
ValidAccuracy = sum(predtValid == tValid)/numel(tValid);
ValidError = (1 - ValidAccuracy)^100;

predtTest = net.classify(xTest);
TestAccuracy = sum(predtTest == tTest)/numel(tTest);
TestError = (1 - TestAccuracy)^100;


temp = string(predtTest);
temp = double(temp);
csvwrite('classifications.csv',temp)
