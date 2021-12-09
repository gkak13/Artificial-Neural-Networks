clc;
clear all;

x1=[ [ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1,...
    -1, -1, -1],[ -1, -1, 1, 1, 1, 1, 1, 1, -1, -1],[ -1, 1, 1, 1, -1,...
    -1, 1, 1, 1, -1],[ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],[ -1, 1, 1, 1,...
    -1, -1, 1, 1, 1, -1],[ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],[ -1, 1, 1,...
    1, -1, -1, 1, 1, 1, -1],[ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],[ -1, 1,...
    1, 1, -1, -1, 1, 1, 1, -1],[ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],[ -1,...
    1, 1, 1, -1, -1, 1, 1, 1, -1],[ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],...
    [ -1, -1, 1, 1, 1, 1, 1, 1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1,...
    -1, -1],[ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1] ];

x2=[ [ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1,...
    -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1,...
    1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1,...
    1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1,...
    -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1,...
    -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1,...
    -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1,...
    1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1,...
    -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1] ];

x3=[ [ 1, 1, 1, 1, 1, 1, 1, 1, -1, -1],[ 1, 1, 1, 1, 1, 1, 1, 1, -1,...
    -1],[ -1, -1, -1, -1, -1, 1, 1, 1, -1, -1],[ -1, -1, -1, -1, -1, 1,...
    1, 1, -1, -1],[ -1, -1, -1, -1, -1, 1, 1, 1, -1, -1],[ -1, -1, -1,...
    -1, -1, 1, 1, 1, -1, -1],[ -1, -1, -1, -1, -1, 1, 1, 1, -1, -1],[ 1,...
    1, 1, 1, 1, 1, 1, 1, -1, -1],[ 1, 1, 1, 1, 1, 1, 1, 1, -1, -1],[ 1,...
    1, 1, -1, -1, -1, -1, -1, -1, -1],[ 1, 1, 1, -1, -1, -1, -1, -1, -1,...
    -1],[ 1, 1, 1, -1, -1, -1, -1, -1, -1, -1],[ 1, 1, 1, -1, -1, -1,...
    -1, -1, -1, -1],[ 1, 1, 1, -1, -1, -1, -1, -1, -1, -1],[ 1, 1, 1, 1,...
    1, 1, 1, 1, -1, -1],[ 1, 1, 1, 1, 1, 1, 1, 1, -1, -1] ];

x4=[ [ -1, -1, 1, 1, 1, 1, 1, 1, -1, -1],[ -1, -1, 1, 1, 1, 1, 1, 1, 1,...
    -1],[ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1,...
    1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],[ -1, -1, -1,...
    -1, -1, -1, 1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],...
    [ -1, -1, 1, 1, 1, 1, 1, 1, -1, -1],[ -1, -1, 1, 1, 1, 1, 1, 1, -1,...
    -1],[ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1,...
    1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],[ -1, -1, -1,...
    -1, -1, -1, 1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],...
    [ -1, -1, 1, 1, 1, 1, 1, 1, 1, -1],[ -1, -1, 1, 1, 1, 1, 1, 1,...
    -1, -1] ];

x5=[ [ -1, 1, 1, -1, -1, -1, -1, 1, 1, -1],[ -1, 1, 1, -1, -1, -1, -1,...
    1, 1, -1],[ -1, 1, 1, -1, -1, -1, -1, 1, 1, -1],[ -1, 1, 1, -1, -1,...
    -1, -1, 1, 1, -1],[ -1, 1, 1, -1, -1, -1, -1, 1, 1, -1],[ -1, 1, 1,...
    -1, -1, -1, -1, 1, 1, -1],[ -1, 1, 1, -1, -1, -1, -1, 1, 1, -1],...
    [ -1, 1, 1, 1, 1, 1, 1, 1, 1, -1],[ -1, 1, 1, 1, 1, 1, 1, 1, 1,...
    -1],[ -1, -1, -1, -1, -1, -1, -1, 1, 1, -1],[ -1, -1, -1, -1, -1,...
    -1, -1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, -1, 1, 1, -1],[ -1, -1,...
    -1, -1, -1, -1, -1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, -1, 1, 1,...
    -1],[ -1, -1, -1, -1, -1, -1, -1, 1, 1, -1],[ -1, -1, -1, -1, -1,...
    -1, -1, 1, 1, -1] ];

fedPattern1 = [[-1, -1, -1, -1, -1, -1, -1, -1, 1, 1], [-1, -1, -1, -1,...
    -1, -1, -1, -1, 1, 1], [1, 1, 1, 1, 1, -1, -1, -1, 1, 1], [1, 1, 1,...
    1, 1, -1, -1, -1, 1, 1], [1, 1, 1, 1, 1, -1, -1, -1, 1, 1], [1, 1,...
    1, 1, 1, -1, -1, -1, 1, 1], [1, 1, 1, 1, 1, -1, -1, -1, 1, 1], [-1,...
    -1, -1, -1, -1, -1, -1, -1, 1, 1], [-1, -1, -1, -1, -1, -1, -1, -1,...
    1, 1], [-1, -1, -1, 1, 1, 1, 1, 1, 1, 1], [-1, -1, -1, 1, 1, 1, 1,...
    1, 1, 1], [-1, -1, -1, 1, 1, 1, 1, 1, 1, 1], [-1, -1, -1, 1, 1, 1,...
    1, 1, 1, 1], [-1, -1, -1, 1, 1, 1, 1, 1, 1, 1], [-1, -1, -1, -1, -1,...
    -1, -1, -1, 1, 1], [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1]];

fedPattern2 = [[-1, -1, -1, 1, 1, 1, 1, -1, -1, -1], [-1, -1, -1, 1, 1,...
    1, 1, -1, -1, -1], [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1], [-1, -1,...
    -1, 1, 1, 1, 1, -1, -1, -1], [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1],...
    [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1], [-1, -1, -1, 1, 1, 1, 1, -1,...
    -1, -1], [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1], [-1, -1, -1, 1, 1, 1,...
    1, -1, -1, -1], [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1], [-1, -1, -1,...
    1, 1, 1, 1, -1, -1, -1], [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1], [-1,...
    -1, -1, 1, 1, 1, 1, -1, -1, -1], [-1, -1, -1, 1, 1, 1, 1, -1, -1,...
    -1], [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1], [1, 1, 1, -1, -1, -1, -1,...
    1, 1, 1]];

fedPattern3 = [[1, 1, 1, -1, -1, 1, -1, 1, 1, -1], [1, 1, 1, -1, -1, 1,...
    -1, 1, 1, -1], [1, 1, 1, -1, -1, 1, -1, 1, 1, -1], [1, 1, 1, -1, -1,...
    1, -1, 1, 1, -1], [1, 1, 1, -1, -1, 1, -1, 1, 1, -1], [1, 1, 1, -1,...
    -1, 1, -1, 1, 1, -1], [1, 1, 1, -1, -1, 1, -1, 1, 1, -1], [1, 1, 1,...
    1, 1, -1, 1, 1, 1, -1], [1, 1, 1, 1, 1, -1, 1, 1, 1, -1], [1, -1,...
    -1, -1, -1, 1, -1, 1, 1, -1], [1, -1, -1, -1, -1, 1, -1, 1, 1, -1],...
    [1, -1, -1, -1, -1, 1, -1, 1, 1, -1], [1, -1, -1, -1, -1, 1, -1, 1,...
    1, -1], [1, -1, -1, -1, -1, 1, -1, 1, 1, -1], [1, -1, -1, -1, -1, 1,...
    -1, 1, 1, -1], [1, -1, -1, -1, -1, 1, -1, 1, 1, -1]];


x1 = x1'; x2 = x2'; x3 = x3'; x4 = x4'; x5 = x5';

fedPattern1 = fedPattern1'; fedPattern2 = fedPattern2';
fedPattern3 = fedPattern3';

patterns = [x1,x2,x3,x4,x5];
feed = [fedPattern1,fedPattern2,fedPattern3];
storedPatterns = size(feed,2); % equal to 3
dim = length(x1);
weightMatrix = zeros(dim);

for i = 1:size(patterns,2)
    
    weightMatrix = weightMatrix + mtimes(patterns,patterns');
    
end

weightMatrix = weightMatrix.*(1/dim);

% diagonal elements set to zero
for i = 1:dim
    for j = 1:i
        if i == j
            weightMatrix(i,j) = 0;
        end
    end
end

loop = zeros(1,storedPatterns);
currentState = zeros(dim,storedPatterns);
for i = 1:storedPatterns
    
    newState = feed(:,i);
    previousState = zeros(dim,1);
    steadyState = 0;
    
    while steadyState == 0
        
        loop(i) = loop(i)+1;
        previousState = newState;
        % asynchronous update
        for k = 1:dim
            
            upd = weightMatrix*newState;
            signum = sign(upd);
            if signum == 0
                signum = 1;
            end
            newState = signum;
            
        end
        %currentState(:,i) = newState;
        
        if newState == previousState
            steadyState = 1;
        end
    
    end
    currentState(:,i) = newState;
   
end

digit = zeros(1,storedPatterns); 
for m = 1:storedPatterns       
    for j = 1:size(patterns,2)     
        if currentState(:,m) == patterns(:,j)
            digit(m) = m;
        elseif currentState(:,m) == -patterns(:,j)
            digit(m) = -m;
        else
            if m == storedPatterns
                digit(m) = 6;
            end
        end
    end
end
    

% visualization of digits



    








