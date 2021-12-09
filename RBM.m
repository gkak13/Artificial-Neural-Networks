%%%%%%%  Kakkos Ioannis Homework 2 - Restricted Boltzmann Machine  %%%%%%%

Npatterns = 8;
Nvisible = 3;
Nhidden = 1;     %  1,2,4,8 -->change this for the number of hidden neurons
x = zeros(Npatterns,3);
% all the possible combinations
x(1,:) = [-1,-1,-1];
x(2,:) = [+1,-1,+1];
x(3,:) = [-1,+1,+1];
x(4,:) = [+1,+1,-1];
x(5,:) = [-1,-1,+1];
x(6,:) = [+1,-1,-1];
x(7,:) = [-1,+1,-1];
x(8,:) = [+1,+1,+1];

visible = zeros(Nvisible,1);
hidden = zeros(Nhidden,1);

weights = randn(Nvisible,Nhidden);
visibleBias = zeros(Nvisible,1);
hiddenBias = zeros(Nhidden,1);
miniBatch = 20;     
k = 100;
eta = 0.1;   % try with 0.1, 0.001

for trials = 1:1000    % not related to k of CD-k  (try with 100,1000)
    
    weightIncrement = zeros(Nvisible,Nhidden);     
    visibleBiasIncrement = zeros(Nvisible,1);    
    hiddenBiasIncrement = zeros(Nhidden,1);      
    
    for mu = 1:miniBatch         % loop over mini batch
        
        indexMu = randi(4);      % random choice of one of the first 4 patterns
        visible = x(indexMu,:)';
        visibleStored = visible;      % store for later use
        localHidden = (visible'*weights)' - hiddenBias;  %  produces an Nhidden-by-1 matrix
        localHiddenProb = zeros(1,Nhidden);
        
        for i = 1:Nhidden
            
            r = rand;
            localHiddenProb(i) = 1/(1+exp(-2*localHidden(i))); % calculate probability of local hidden  
            if r < localHiddenProb(i)                          % stochastic update- p(x)=1/(1+exp(-2*x))
                hidden(i) = +1;
            else
                hidden(i) = -1;
            end
            
        end       
        
        for j = 1:k
            % update visible neurons
            localVisible = weights*hidden - visibleBias;   % produces an Nvisible-by-1 matrix
            localVisibleProb = zeros(1,Nvisible);
            % perform stochastic update
            for v = 1:Nvisible
                
                r = rand;
                localVisibleProb(v) = 1/(1+exp(-2*localVisible(v)));
                if r < localVisibleProb(v)
                    visible(v) = +1;
                else
                    visible(v) = -1;
                end
                
            end
            
            localHidden = (visible'*weights)' - hiddenBias;
            
            for h = 1:Nhidden
                
                r = rand;
                localHiddenProb(h) = 1/(1+exp(-2*localHidden(h)));
                if r < localHiddenProb(h)
                    hidden(h) = +1;
                else
                    hidden(h) = -1;
                end
                
            end
            
        end      
        
        localHiddenStored = (visibleStored'*weights)' - hiddenBias;
        weightIncrement = weightIncrement + eta*(visibleStored*(tanh(localHiddenStored))'-visible*(tanh(localHidden))');    % should be an outter product ?W update
        visibleBiasIncrement = visibleBiasIncrement - eta*(visibleStored - visible);
        hiddenBiasIncrement = hiddenBiasIncrement - eta*(tanh(localHiddenStored)-tanh(localHidden));
        
    end    % end of mu loop   (mini Batch loop)
    
    
    weights = weights + weightIncrement;  %   update the weights
    visibleBias = visibleBias + visibleBiasIncrement;
    hiddenBias = hiddenBias + hiddenBiasIncrement;

end     %  end trials

%%%%%%%%%% THS IS THE END OF TRAINING   %%%%%%%%%%%%%%%
%%%%    COMPUTE KL DIVERGENCE                         %%%%%%%%%%%%%
%%%%    RUNNING THE BOLTZMANN MACHINE A LOT OF TIMES  %%%%%%%%%%%%%

Noutter = 2000;         % this can be any large value
Ninner = 1000;          % this can be any large value  
PBoltzmann = zeros(Npatterns,1);
freq = zeros(Npatterns,1);

for i = 1:Noutter
    
    Mu = randi(8);
    visible = x(Mu,:)';
    
    localHidden = (visible'*weights)' - hiddenBias;
    for h = 1:Nhidden
            r = rand;
            localHiddenProb(h) = 1/(1+exp(-2*localHidden(h)));
            if r < localHiddenProb(h)
                hidden(h) = +1;
            else
                hidden(h) = -1;
            end
    end
    
    for j = 1:Ninner
        
        localVisible = weights*hidden - visibleBias;
        for v = 1:Nvisible
            r = rand;
            localVisibleProb(v) = 1/(1+exp(-2*localVisible(v)));
            if r < localVisibleProb(v)
                visible(v) = +1;
            else
                visible(v) = -1;
            end                
        end
        
        localHidden = (visible'*weights)' - hiddenBias;
        for h = 1:Nhidden
            r = rand;
            localHiddenProb(h) = 1/(1+exp(-2*localHidden(h)));
            if r < localHiddenProb(h)
                hidden(h) = +1;
            else
                hidden(h) = -1;
            end
        end
        
        for l = 1:Npatterns
            if visible == x(l)
                freq(l) = freq(l) + 1;    % observing the appearence frequency of each of the four patterns
                PBoltzmann(l) = PBoltzmann(l) + 1/(Noutter*Ninner);
            end 
        end
         
    end

end

% Calculate Kullback-Leibler divergence
Pdata = zeros(Npatterns,1);
for i = 1:4
    Pdata(i) = 1/4;
end
for i = 5:8
    Pdata(i) = 0;
end
    
KL = 0;
for i = 1:Npatterns
    
    if Pdata(i) ~= 0
        KL = KL + Pdata(i)*log10(Pdata(i)/PBoltzmann(i)); 
    end
  
end




