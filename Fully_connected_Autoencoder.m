%% Artificial Neural Networks     %%
%% Kakkos Ioannis - 9304136030    %%
%% Fully connected Autoencoder    %%
% clear all;

%% Data loading
[xTrain,tTrain,xVal,tVal,xTest,tTest] = LoadMNIST(3);

xTrain = reshape(xTrain,28*28,50000);
xTrain = double(xTrain);
xTrain = xTrain/255;

xTest = reshape(xTest,28*28,10000);
xTest = double(xTest);
xTest = xTest/255;

xVal = reshape(xVal,28*28,10000);
xVal = double(xVal);
xVal = xVal/255;

%% Autoencoder 1
layers = [
    sequenceInputLayer(784),...
    fullyConnectedLayer(50,'WeightsInitializer','glorot'),...
    reluLayer,...
    fullyConnectedLayer(2,'WeightsInitializer','glorot'),...
    reluLayer,...
    fullyConnectedLayer(784,'WeightsInitializer','glorot'),...
    reluLayer,...
    regressionLayer];
      
options = trainingOptions('adam',...
           'MiniBatchSize',8192,...
           'MaxEpochs',920,...
           'InitialLearnRate',0.001,...
           'ExecutionEnvironment','gpu',...
           'Plots','training-progress');
       
[net1,tr1] = trainNetwork(xTrain,xTrain,layers,options);
bottleneck = net1.Layers(4);
regression = net1.Layers(8);

% Encoder
encodeL(1) = net1.Layers(1);
encodeL(2) = net1.Layers(2);
encodeL(3) = net1.Layers(3);
encodeL(4) = bottleneck;
encodeL(5) = regression;
encodeNet = assembleNetwork(encodeL);

% Decoder
decodeL(1) = sequenceInputLayer(2);
decodeL(2) = net1.Layers(5);
decodeL(3) = net1.Layers(6);
decodeL(4) = net1.Layers(7);
decodeL(5) = regression;
decodeNet = assembleNetwork(decodeL);

% Validate
testPrediction = net1.predict(xTest);
left = reshape(xTest,28,28,10000);
right = reshape(testPrediction,28,28,10000);
scatterT = encodeNet.predict(xTest(:,1:1000));

%% Autoencoder 2
layers2 = [
    sequenceInputLayer(784),...
    fullyConnectedLayer(50,'WeightsInitializer','glorot'),...
    reluLayer,...
    fullyConnectedLayer(4,'WeightsInitializer','glorot'),...
    reluLayer,...
    fullyConnectedLayer(784,'WeightsInitializer','glorot'),...
    reluLayer,...
    regressionLayer];

options = trainingOptions('adam',...
           'MiniBatchSize', 8192,...
           'MaxEpochs', 920,...
           'InitialLearnRate',0.001,...
           'ExecutionEnvironment','gpu');

[net2,tr2] = trainNetwork(xTrain,xTrain,layers2,options);
bottleneck2 = net2.Layers(4);
% regression2 = net2.Layers(8);

% Encoder
encodeL2(1) = net2.Layers(1);
encodeL2(2) = net2.Layers(2);
encodeL2(3) = net2.Layers(3);
encodeL2(4) = bottleneck2;
encodeL2(5) = regression;
encodeNet2 = assembleNetwork(encodeL2);

% Decoder
decodeL2(1) = sequenceInputLayer(4);
decodeL2(2) = net2.Layers(5);
decodeL2(3) = net2.Layers(6);
decodeL2(4) = net2.Layers(7);
decodeL2(5) = regression;
decodeNet2 = assembleNetwork(decodeL2);

%% Validation
testPrediction2 = net2.predict(xTest);
right2 = reshape(testPrediction2,28,28,10000);

%% Plots
plotLeft = [left(:,:,189) left(:,:,6) left(:,:,44) left(:,:,69)...
    left(:,:,66) left(:,:,103) left(:,:,99) left(:,:,98) left(:,:,135)...
    left(:,:,126)];
plotRight1 = [right(:,:,189) right(:,:,6) right(:,:,44) right(:,:,69)...
    right(:,:,66) right(:,:,103) right(:,:,99) right(:,:,98)...
    right(:,:,135) right(:,:,126)];
plotRight2 = [right2(:,:,189) right2(:,:,6) right2(:,:,44) right2(:,:,69)...
    right2(:,:,66) right2(:,:,103) right2(:,:,99) right2(:,:,98)...
    right2(:,:,135) right2(:,:,126)];
plotEnsemble = vertcat(plotLeft, plotRight1, plotRight2);

figure;
hold on;
title('Original vs Produced','Interpreter','Latex');
montage(plotEnsemble,'size',[1 NaN]);
hold off;

figure;
hold on;
title('Scatter','Interpreter','Latex');
scatterRight(scatterT,tTest);
hold off;

a = 12; b = 4;
pt1 = [a/3; a; a/3; a/3]; pt2 = [b; b/4; b; b];

p1 = decodeNet2.predict(pt1);
p2 = decodeNet2.predict(pt2);

imshow(reshape(p1,28,28));
imshow(reshape(p2,28,28));

function scatterRight(scatterT,tTest)
sc0 = zeros(2,1);
sc1 = zeros(2,1);
sc2 = zeros(2,1);
sc3 = zeros(2,1);
sc4 = zeros(2,1);
sc5 = zeros(2,1);
sc6 = zeros(2,1);
sc7 = zeros(2,1);
sc8 = zeros(2,1);
sc9 = zeros(2,1);

tTest = grp2idx(tTest);
for t = 1:length(scatterT)
    dif = tTest(t) - 1;
    if dif == 0
        sc0 = [sc0 scatterT(:,t)];
    elseif diff == 1
        sc1 = [sc1 scatterT(:,t)];
    elseif diff == 2
        sc2 = [sc2 scatterT(:,t)];
    elseif diff == 3
        sc3 = [sc3 scatterT(:,t)];
    elseif diff == 4
        sc4 = [sc4 scatterT(:,t)];
    elseif diff == 5
        sc5 = [sc5 scatterT(:,t)];
    elseif diff == 6
        sc6 = [sc6 scatterT(:,t)];
    elseif diff == 7
        sc7 = [sc7 scatterT(:,t)];
    elseif diff == 8
        sc8 = [sc8 scatterT(:,t)];
    elseif diff == 9
        sc9 = [sc9 scatterT(:,t)];
    end
end

scatter(sc0(1,:),sc0(2,:),'x');
hold on;
scatter(sc1(1,:),sc1(2,:),'+');
hold on;
scatter(sc2(1,:),sc2(2,:),'*');
hold on;
scatter(sc3(1,:),sc3(2,:),'^');
hold on;
scatter(sc4(1,:),sc4(2,:),'d');
hold on;
scatter(sc5(1,:),sc5(2,:),'o');
hold on;
scatter(sc6(1,:),sc6(2,:),'.');
hold on;
scatter(sc7(1,:),sc7(2,:),'s');
hold on;
scatter(sc8(1,:),sc8(2,:),'p');
hold on;
scatter(sc9(1,:),sc9(2,:),'h');
hold off;
legend('0','1','2','3','4','5','6','7','8','9');
end



