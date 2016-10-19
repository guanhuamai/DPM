function [model, decisionMatrix] = iohmmTraining(observations, inputObs, nstates, ostates, numiter)
    model = initializeHMMmodel('discrete', nstates, ostates, 2);
    [model, decisionMatrix] = IOhmmFit(observations, inputObs, model, numiter);
end