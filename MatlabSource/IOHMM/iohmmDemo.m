
function model = iohmmDemo( )
    clc
    model.type = 'discrete';
    
    %% Demo for discrete case
    if strcmp(model.type,'discrete')
        model.nstates = 3; 
        model.ostates = 2;
        model.inputDimension = 2;
        model.B =[1/6  5/6;
                 3/6   3/6;
                 3/7   4/7];

        model.W = [[0.5,0.5], [0.5,0.5];
                  [0.4,0.7], [0.6,0.4];
                  [0.1,0.9], [0.1,0.9]];
        model.piW = [[0.4; 0.6], [0.2; 0.8]]';
        model.type = 'discrete';
        T = 10;
        N = 100;
        
        [observations,hidden_states,inputObs] = hmmSample(model,T,N);
        model = initializeHMMmodel(model.type,model.nstates,model.ostates,model.inputDimension);
        [model, alph] = IOhmmFit(observations,inputObs,model,300);
        disp(alph)
        disp(model.B)
    elseif strcmp(model.type,'gauss')
        model.nstates = 4; 
        model.observationDimension = 10;
        model.A = [0.6 0.15 0.20 0.05;
                  0.10 0.70 0.15 0.05;
                  0.10 0.30 0.10 0.50;
                  0.30 0.10 0.30 0.30];
        for i = 1:model.nstates
            model.mu{i} = rand(model.observationDimension,1);
            model.sigma{i} = rbfK(rand(),model.observationDimension);
        end;
        model.pi = [0.8 0.1 0.1 0]';
        T = 10;
        N = 10;
        [observations,hidden_states] = hmmSample(model,T,N);
        model = initializeHMMmodel(model.type,model.nstates,model.observationDimension);
        
        model.prior.use = 1; % 1 use the prior, 0 not prior on parameters
        %% Prior values
        model.prior.k0 = 1;
        model.prior.mu0 = (1.0/model.observationDimension)*ones(model.observationDimension,1);
        model.prior.Psi = (1.0/model.observationDimension)*eye(model.observationDimension,model.observationDimension);
        truemodel = model;        
        hmmFit(truemodel,observations,model,300);
    end;

end

function K = rbfK(gam,s)
    K = zeros(s,s);
    for i = 1:s
        for j = 1:s
            K(i,j) = normpdf(i-j,0,gam);
        end;
    end;
    
end

