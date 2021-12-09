%% Chaotic time series prediction %%

% training_set = csvread('training-set.csv');
% test_set = csvread('test-set-2.csv');
training_set = readtable('training-set.csv');
test_set = readtable('test-set-2.csv');

x_Train = training_set;
x_Test = test_set;

%% Variables
N = 500; % number of neurons in the reservoir
timesteps = length(x_Train);
std_in = sqrt(0.002);
std_res = sqrt(2/500);
k = 0.01; % ridge parameter

%% Initializing weight matrices and reservoir states
weights_in = std_in.*randn(N,3);
weights = std_res.*randn(N);
weights_out = zeros(3,N);
R = zeros(N,timesteps);
r = zeros(1,timesteps);

%% Run training dynamics
for t = 1:timesteps
R(:,t) = r(t);    
r(t + 1) = tanh(weights*r(t) + weights_in*x_Train(t));
O(t) = weights_out*r(t);
end





