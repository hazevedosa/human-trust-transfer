clear all; close all; clc;


%% Generate new fixed or random task orders

FixedOrRandom = true;

if FixedOrRandom
    load('Randomizers.mat');
else
    % Randomizer
    randomizer = randperm(50);

    % Vector that originates the performances

    V = ones(5, 10);
    V(5,:) = -1 * V(5,:);
    
    % flipping the sign to make t1 = t2 = t3 = t4 = 1 - t5 = 0.9
    
    VecPerformances = [];
    for i = 1:5
        n = ceil(10 * rand);
        V(i, n) = -1 * V(i, n);
        
        VecPerformances = [VecPerformances, V(i, :)];
    end

    % Observed tasks
    obsTasks = zeros(6, 50);
    obsTasks(2:end, :) = 100 * rand(5, 50);

end

%% 


VecTasks = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, ...
            3, 3, 3, 3, 3, 3, 3, 3, 3, 3, ...
            4, 4, 4, 4, 4, 4, 4, 4, 4, 4, ...
            5, 5, 5, 5, 5, 5, 5, 5, 5, 5];

stackVec = [VecTasks;
            VecPerformances];
        



stackVec(:, randomizer) = stackVec;

Trusts = 0.5 * ones(50, 5);
PerfSums = zeros(1, 5);
Counters = zeros(1, 5);

for i = 1:50
    for j = 1:5
        if stackVec(1,i) == j
            Counters(j) = Counters(j) + 1;
            PerfSums(j) = PerfSums(j) + stackVec(2,i);
            Trusts(i, j) = 0.5 *(1 + PerfSums(j)/Counters(j));
        end
    end
    if i ~= 50
        Trusts(i+1,:) = Trusts(i,:);
    end
end

% build matrices
matTasks = zeros(50, 50);
matPerformances = zeros(50, 50);


for i = 1:50
    for j = 1:i
        matTasks(i, 51 - j) = stackVec(1, i - j + 1);
        matPerformances(i, 51 - j) = stackVec(2, i - j + 1);
    end
end



obsTasksStruct = zeros(50, 50, 50);

for i = 1:50
    for j = 1:50
        taskID = matTasks(i, j);
        obsTasksStruct(i, j, :) = obsTasks(taskID + 1, :);
    end
end
    

for i = 1:50
    for j = 1:50
        taskRawPerf = matPerformances(i, j);
        if taskRawPerf == 0
            matSuccess(i, j) = 0;
            matFail(i, j) = 0;
        elseif taskRawPerf == 1
            matSuccess(i, j) = 1;
            matFail(i, j) = 0;
        else
            matSuccess(i, j) = 0;
            matFail(i, j) = 1;            
        end
        
    end
end

matFail = cat(2, matFail, matFail, matFail, matFail, matFail);
matSuccess = cat(2, matSuccess, matSuccess, matSuccess, matSuccess, matSuccess);



tasksobsfeats = cat(2, obsTasksStruct, obsTasksStruct, obsTasksStruct, obsTasksStruct, obsTasksStruct);

tasksobsperf = cat(3, matFail, matSuccess);

taskspredfeats = repelem(obsTasks(2:end, :), 50, 1);

trustpred = [Trusts(:, 1); Trusts(:, 2); Trusts(:, 3); Trusts(:, 4); Trusts(:, 5)];

tasksobsids(:,:,1) = cat(2, matTasks, matTasks, matTasks, matTasks, matTasks);

taskpredids = repelem([1 2 3 4 5]', 50, 1);

taskpredtrust = trustpred;


%% Save...

save(...
    '~/Documents/human-trust-transfer/code/RawDataset.mat',...
                                           'tasksobsfeats',...
                                            'tasksobsperf',...
                                          'taskspredfeats',...
                                               'trustpred',...
                                             'tasksobsids',...
                                             'taskpredids',...
                                            'taskpredtrust'...
    )
    
    

