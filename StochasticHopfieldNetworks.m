%%%% Stochastic Hopfield Network %%%%
%%%% Artificial Neural Networks %%%%
%%%% Homework 1 %%%%
%%%% Kakkos Ioannis %%%%

clear all; clc;
%% Given data
neurons = 200;        
beta = 2;                   % noise parameter
patterns = [7 45];          % try for 7 and 45 patterns
updatesT = 2*(10^5);       % number of asynchronous stochastic updates
experimentRepeat = 100;     % times to repeat the experiment

%% Initialization
meanOrderParameter = zeros(1,length(patterns));
OPoverTrials = zeros(1,updatesT);
OPoverExp = zeros(1,experimentRepeat);
totalOP = zeros(1,length(patterns));

%% Loop
for p = 1:length(patterns)    
for i = 1:experimentRepeat
% Constructing states matrix
states = randi([-1 0], neurons, patterns(p));
for var1 = 1:neurons
    for var2 = 1:patterns(p)
        if states(var1,var2) == 0 
            states(var1,var2) = 1;
        end
    end
end

% Constructing weight matrix
weights = zeros(neurons);
for j = 1:patterns(p)
    weights = weights + states(:,j)*states(:,j)'; 
end
weights = weights.*(1/neurons);
for var1 = 1:neurons
    for var2 = 1:var1
        if var1 == var2
            weights(var1,var2) = 0;
        end
    end
end

% Feed pattern number 1
fed = states(:,1);

% Perform asynchronous update
for t = 1:updatesT

% Picking a random neuron in the weight matrix to perform the update
pickNeuron = randi(neurons);

% Perform stochastic update
localField = weights(pickNeuron,:)*fed;
localProbability = 1/(1+exp(-2*beta*localField));
r = rand;
if localProbability > r    
    currentState = +1;
else
    currentState = -1; 
end
fed(pickNeuron) = currentState;
% Order Parameter (OP) over trials calculation
OPoverTrials(t) = (1/neurons)*(fed'*states(:,1));
end
% Order parameter over experiment calculation
OPoverExp(i) = mean(OPoverTrials);
end
% Total order parameter
totalOP(p) = mean(OPoverExp);
end

disp('The overall mean order parameter for 7 patterns is:')
totalOP(1)
disp('and for 45 patterns:')
totalOP(2)




