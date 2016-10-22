function decisionMatrix = iohmmTraining(observations, inputObs, nstates, ostates, numiter)
    for i=1:size(observations, 2)
         observations{i} = real(cell2mat(observations{i})) + 1;
    end;
    for i=1:size(inputObs, 2)
        for j=1:size(inputObs{i}, 2)
            inputObs{i}{j} = double(cell2mat(inputObs{i}{j}'));
        end;
        inputObs{i} = cell2mat(inputObs{i});
    end;
    disp('start training iohmm')
    nstates = double(nstates);
    ostates = double(ostates);
    numiter = double(numiter);
    model = initializeHMMmodel('discrete', nstates, ostates, 2);
    decisionMatrix = IOhmmFit(observations, inputObs, model, numiter);
end