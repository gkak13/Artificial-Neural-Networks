

clear all;clc;
%% Data loading
trainingSet = csvread("training_set.csv");
validationSet = csvread("validation_set.csv");
trainingInput1 = normalize(trainingSet(:,1));
trainingInput2 = normalize(trainingSet(:,2));
trainingTarget = trainingSet(:,3);
validationInput1 = normalize(validationSet(:,1));
validationInput2 = normalize(validationSet(:,2));
validationTarget = validationSet(:,3);

%% Normalisation
newTrainingInput = [trainingInput1,trainingInput2];
newValidationInput = [validationInput1,validationInput2];
%% Parameterisation and initialisation
M1 = 5;
C = inf;
eta = 0.005;                    % step-length
locate = length(trainingSet);
threshold1 = zeros(M1,1);
threshold2 = 0;
weights1 = randn([M1,width(newTrainingInput)]);
weights2 = randn([M1, 1]);
visible1 = zeros(M1,1);
visible2 = 0;
counting = 0;

%% Main loop
while C >= 0.12
    counting = counting + 1;
    % Random choice of mu and pattern application to input layer
    mu = randi(locate);
    visible1 = newTrainingInput(mu,:);
    
    % Forward propagation
    for j = 1:M1
        visible2(j) = tanh(sum(newTrainingInput(mu,:).*weights1(j,:) -...
            threshold1(j)));
    end
    
    % Compute output
    output = tanh(sum(weights2.*visible2') - threshold2);
    
    % Output layer errors
    errors1 = (trainingTarget(mu) - output)*(1 -...
    (tanh(dot(weights2,visible2) - threshold2)^2));
    gPrime = 1 - tanh(weights1*visible1' - threshold1).^2;
    %errors1 = errors1(:,1);
    for k = 1:M1
        errors2(k) = errors1*weights2(k)*gPrime(k);
    end
    
    % Weights and thresholds update 
    weightIncrement1 = eta*errors2'*visible1;
    weightIncrement2 = eta*errors1'*visible2;
    thresholdIncrement1 = eta*errors2';
    thresholdIncrement2 = eta*errors1;
    weights1 = weights1 + weightIncrement1;
    weights2 = weights2 + (weightIncrement2)';
    threshold1 = threshold1 - thresholdIncrement1;
    threshold2 = threshold2 - thresholdIncrement2;
    
    % Calculate classification error
    if rem(counting,1000) == 0
    nominatorC = 0;
    pVal = length(validationSet);
    for var1 = 1:pVal
        visible1 = newValidationInput(var1,:); 
        for var2 = 1:M1
            visible2(var2) = tanh(sum(newValidationInput(var1,:).*...
                weights1(var2,:)) - threshold1(var2));
        end
        output = tanh(sum(weights2.*visible2') - threshold2);
        nominatorC = nominatorC + abs(sign(output) -...
        validationTarget(var1));
    end
    C = nominatorC/(2*pVal);
    if C < 0.116
        disp('Found!')
        w1 = weights1;
        w2 = weights2;
        t1 = threshold1;
        t2 = threshold2;
        csvwrite('w1.csv',w1);
        csvwrite('w2.csv',w2);
        csvwrite('t1.csv',t1);
        csvwrite('t2.csv',t2);
        break
    end
    disp(C)
end
end





