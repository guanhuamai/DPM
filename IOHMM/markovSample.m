


function S = markovSample(model, len, nsamples)
% Sample from a markov distribution
% model has fields pi, A as returned by markovFit
%
% S is of size nsamples-by-len
%

% This file is from pmtk3.googlecode.com

if nargin < 3, nsamples = 1; end
piW = model.piW;
W = model.W;
inputObs = model.inputObs;
S = zeros(nsamples, len);
for i=1:nsamples
    prob = multiClassProbability(piW, inputObs{i}(:, 1));
    S(i, 1) = sampleDiscrete(prob);
    for t=2:len
        A = transitionMatrix(model.W,inputObs{i}(:,t),model); 
        S(i, t) = sampleDiscrete(A(S(i, t-1), :));
    end
end
end


function probability = multiClassProbability(W,U)
    %% Returns probability of each class Softmax regression
    potentials = [exp(W*U);1.0];
    Z = sum(potentials);
    probability = potentials./Z;
end

function A = transitionMatrix(W,U,model)
    %% Returns the state transition matrix for the given input U
    A = zeros(model.nstates,model.nstates);
    for i = 1:model.nstates
        probability = multiClassProbability(reshape(W(i,:,:),model.nstates-1,model.inputDimension),U);
        A(i,:) = probability';
    end;
end
