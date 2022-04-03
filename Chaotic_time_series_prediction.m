%% Artificial Neural Networks         %%
%% Kakkos Ioannis                     %%
%% Chaotic time series prediction     %%

training_set = csvread('training-set.csv');
test_set = csvread('test-set-2.csv');

x_Train = training_set;
x_Test = test_set;

%% Variables
N = 500; % number of neurons in the reservoir
trainLen = length(x_Train(1,:));
testLen = length(x_Test(1,:));
std_in = sqrt(0.002);
std_res = sqrt(2/500);
k = 0.01; % ridge parameter

%% Initializing weight matrices and reservoir states
weights_in = randn(N,3).*std_in;
weights = randn(N,N).*std_res;
R = zeros(N,trainLen);
r = zeros(N,1);

%% Run training dynamics
for t = 1:trainLen
    R(:,t) = r;
    r = tanh(weights*r + weights_in*x_Train(:,t));
end

% Construct weights out matrix from the training dynamics
weights_out = x_Train*R'*inv(R*R' + k*eye(N));

%% Prediction dynamics
R_pred = zeros(N,testLen);
r_pred = zeros(N,1);
for t = 1:testLen
    R_pred(:,t) = r_pred;
    r_pred = tanh(weights*r_pred + weights_in*x_Test(:,t));
end

r_pred = R_pred(:,testLen);
O = zeros(N,1);
X = zeros(N,1);
Y = zeros(N,1);

for i = 1:N
    temp = weights_out*r_pred;
    X(i) = temp(1);
    O(i) = temp(2);
    Y(i) = temp(3);
    r_pred = tanh(weights*r_pred + weights_in*temp);
end

%% Export csv file and plot
csvwrite('prediction.csv',O);

figure;
hold on;
title('Chaotic Lorentz dynamics','Interpreter','Latex');
xlabel('x','Interpreter','Latex');
ylabel('y','Interpreter','Latex');
zlabel('z','Interpreter','Latex');
plot3(X,O,Y,'b')
hold off;
