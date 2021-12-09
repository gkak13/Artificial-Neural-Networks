clc;
clear all;


%Kakkos-Ioannis-hw1-problem1
p = [12; 24; 48; 70; 100; 120];   
N = 120;                          
independentTrials = 10^5;
errorCounter = zeros(1,length(p));

for k = 1:length(p)
    
    loopCounter = 0;
    while loopCounter <= independentTrials
    
        loopCounter = loopCounter +1;
        
        patternsMatrix = zeros(N,p(k));          
        for i = 1:N
            for j = 1:p(k)
                 patternsMatrix(i,j) = randi(2); % equally possible values
                 if patternsMatrix(i,j) == 2
                     patternsMatrix(i,j) = -1;
                 end
            end
        end
    
        weightMatrix = zeros(N,N);
    
        for i = 1:p(k)
            weightMatrix = weightMatrix + mtimes(patternsMatrix,...
            patternsMatrix.');
        end
        weightMatrix = weightMatrix.*(1/N);
    
        % defining zero diagonal elements
%         for i = 1:N
%             for j = 1:i
%                 if i == j
%                     weightMatrix(i,j) = 0;
%                 end
%             end
%         end
    
        % random selection of pattern and neuron
        randomNeuron = randi(N);
        randomPattern = randi(p(k));
        
        % performing the update
        upd = weightMatrix(randomNeuron,:)*patternsMatrix(:,randomPattern);
        signum = sign(upd);
        if signum == 0
            signum = 1;
        end
        
        % error checking
        if signum ~= patternsMatrix(randomNeuron,randomPattern)
            errorCounter(k) = errorCounter(k) + 1;
        end
        
    end
        
end 

% error calculation
totalError = errorCounter./independentTrials;
disp(totalError)



        
        