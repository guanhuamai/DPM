function [observations,hidden_states,inputObs] = hmmSample( model,T,Nobs )

    model.inputDimension = 2;
    model.inputObs = cell(1, Nobs);
    for i = 1: Nobs
        %model.inputObs{i} = cell(model.inputDimension, T);
        model.inputObs{i} = sampleDiscrete([0.5, 0.5], 1, T) - 1;
        model.inputObs{i} = repmat(model.inputObs{i}, 2, 1);
        model.inputObs{i}(2,:) = 1 - model.inputObs{i}(1,:)
    end;
    inputObs = model.inputObs;
    S = markovSample(model, T, Nobs);
    observations = cell(1,Nobs);
    hidden_states = cell(1,Nobs);


    if strcmp(model.type,'discrete')
        for i = 1:Nobs
            hidden_states{i} = S(i,:);
            observations{i} = zeros(1, T);
            for t=1:T
                observations{i}(t) = sampleDiscrete(model.B(hidden_states{i}(t), :));
            end;
        end;
    elseif strcmp(model.type,'gauss')
        for i = 1:Nobs
            hidden_states{i} = S(i,:);
            observations{i} = zeros(model.observationDimension, T);
            for t=1:T
                k = hidden_states{i}(t); 
                observations{i}(:, t) = colvec(gaussSample(model.mu{k}, model.sigma{k}, 1));
            end
        end;        
    end;



end

