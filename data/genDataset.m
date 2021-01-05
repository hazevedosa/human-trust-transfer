clear all; close all;  clc;

raw_table = readtable('qualtricsRawData.xlsx');
raw_table(1,:) = []; % if using macOS

num_responses = size(raw_table, 1);


obs_task_seq = [];
pred_task = [];
obs_task_sens_cap_seq = [];
obs_task_proc_cap_seq = [];
pred_task_sens_cap = [];
pred_task_proc_cap = [];
trust_pred = [];




for i = 1:num_responses
    
    participant_data = raw_table(i, :);
    participant_prediction_task = str2num(participant_data.randNumber{1});
    participant_videos_order_raw = participant_data.FL_15_DO{1};
    participant_videos_order = [str2num(participant_videos_order_raw(19)), ...
                                str2num(participant_videos_order_raw(39)), ...
                                str2num(participant_videos_order_raw(59)), ...
                                str2num(participant_videos_order_raw(79))];
                            
    participant_videos_order = setdiff(participant_videos_order, participant_prediction_task, 'stable');
    
    participant_fail_succ = [str2num(participant_data.fail_succ_1{1}), ...
                             str2num(participant_data.fail_succ_2{1}), ...
                             str2num(participant_data.fail_succ_3{1}), ...
                             str2num(participant_data.fail_succ_4{1})];
    
    %% sensing check
    sensing_diff_order = [str2num(participant_data.Sens_1_1{1}), ...
                          str2num(participant_data.Sens_1_2{1}), ...
                          str2num(participant_data.Sens_1_3{1}), ...
                          str2num(participant_data.Sens_1_4{1})];
    
    sensing_capabilities = [str2num(participant_data.Sens_2_1{1}), ...
                            str2num(participant_data.Sens_2_2{1}), ...
                            str2num(participant_data.Sens_2_3{1}), ...
                            str2num(participant_data.Sens_2_4{1})];
                        
    [~, sensing_diff_order_from_capabilities] = sort(sensing_capabilities);
    
    
    if any(sensing_diff_order - sensing_diff_order_from_capabilities) ~= 0
%         disp(i);
%         disp('Warning in Sensing -- possibly need to remove participant.');
    end    
    
    %% processing check
    processing_diff_order = [str2num(participant_data.Proc_1_1{1}), ...
                             str2num(participant_data.Proc_1_2{1}), ...
                             str2num(participant_data.Proc_1_3{1}), ...
                             str2num(participant_data.Proc_1_4{1})];
    
    processing_capabilities = [str2num(participant_data.Proc_2_1{1}), ...
                               str2num(participant_data.Proc_2_2{1}), ...
                               str2num(participant_data.Proc_2_3{1}), ...
                               str2num(participant_data.Proc_2_4{1})];
                        
    [~, processing_diff_order_from_capabilities] = sort(processing_capabilities);
    
    
    if any(processing_diff_order - processing_diff_order_from_capabilities) ~= 0
%         disp(i);
%         disp('Warning in Processing -- possibly need to remove participant.');
    end

    participant_observed_task = [[0, 0, participant_videos_order(1)];
                                 [0, participant_videos_order(1), participant_videos_order(2)];
                                 [participant_videos_order(1), participant_videos_order(2), participant_videos_order(3)]];
                        
    participant_sensing_capabilities = [[0, 0, sensing_capabilities(participant_videos_order(1))];
                                        [0, sensing_capabilities(participant_videos_order(1)), sensing_capabilities(participant_videos_order(2))];
                                        [sensing_capabilities(participant_videos_order(1)), sensing_capabilities(participant_videos_order(2)), sensing_capabilities(participant_videos_order(3))]];
    
    participant_sens_cap_pred_task = [sensing_capabilities(participant_prediction_task);
                                      sensing_capabilities(participant_prediction_task);
                                      sensing_capabilities(participant_prediction_task)];
                                    
                                    
    participant_processing_capabilities = [[0, 0, processing_capabilities(participant_videos_order(1))];
                                           [0, processing_capabilities(participant_videos_order(1)), processing_capabilities(participant_videos_order(2))];
                                           [processing_capabilities(participant_videos_order(1)), processing_capabilities(participant_videos_order(2)), processing_capabilities(participant_videos_order(3))]];

    participant_proc_cap_pred_task = [processing_capabilities(participant_prediction_task);
                                      processing_capabilities(participant_prediction_task);
                                      processing_capabilities(participant_prediction_task)];
                                       
    participant_performances = zeros(3,3,2);
                                       
    participant_performances(:, :, 1) = [[0, 0, not(participant_fail_succ(participant_videos_order(1)))];
                                         [0, not(participant_fail_succ(participant_videos_order(1))), not(participant_fail_succ(participant_videos_order(2)))];
                                         [not(participant_fail_succ(participant_videos_order(1))), not(participant_fail_succ(participant_videos_order(2))), not(participant_fail_succ(participant_videos_order(3)))]];

    participant_performances(:, :, 2) = [[0, 0, participant_fail_succ(participant_videos_order(1))];
                                         [0, participant_fail_succ(participant_videos_order(1)), participant_fail_succ(participant_videos_order(2))];
                                         [participant_fail_succ(participant_videos_order(1)), participant_fail_succ(participant_videos_order(2)), participant_fail_succ(participant_videos_order(3))]];


    participant_trust_raw = {raw_table(i, 46:50);
                             raw_table(i, 51:55);
                             raw_table(i, 56:60);
                             raw_table(i, 61:65)};

    participant_prediction_task = [participant_prediction_task;
                                   participant_prediction_task;
                                   participant_prediction_task];    


    participant_trust_1 = participant_trust_raw{participant_videos_order(1)};
    
    for j = 1:2
        if strcmp(participant_trust_1{1, j}{1}, 'Yes')
            attChk_1 = 1;
        elseif strcmp(participant_trust_1{1, j}{1}, 'No')
            attChk_1 = 0;
        end
    end
    
    if attChk_1 == participant_fail_succ(participant_videos_order(1))
%         disp(i);
%         disp('Att Chk 1 OK');
    else
        disp(i);
        disp('Att Chk 1 NOK');
    end
    
    for j = 3:5
        if ~strcmp(participant_trust_1{1, j}{1}, '')
            trust_prediction_1 = str2num(participant_trust_1{1, j}{1});
        end
    end
    
    
    
    participant_trust_2 = participant_trust_raw{participant_videos_order(2)};
    
    for j = 1:2
        if strcmp(participant_trust_2{1, j}{1}, 'Yes')
            attChk_2 = 1;
        elseif strcmp(participant_trust_2{1, j}{1}, 'No')
            attChk_2 = 0;
        end
    end
    
    if attChk_2 == participant_fail_succ(participant_videos_order(2))
%         disp(i);
%         disp('Att Chk 2 OK');
    else
        disp(i);
        disp('Att Chk 2 NOK');
    end
    
    for j = 3:5
        if ~strcmp(participant_trust_2{1, j}{1}, '')
            trust_prediction_2 = str2num(participant_trust_2{1, j}{1});
        end
    end

    
    participant_trust_3 = participant_trust_raw{participant_videos_order(3)};

    
    for j = 1:2
        if strcmp(participant_trust_3{1, j}{1}, 'Yes')
            attChk_3 = 1;
        elseif strcmp(participant_trust_3{1, j}{1}, 'No')
            attChk_3 = 0;
        end
    end
    
    if attChk_3 == participant_fail_succ(participant_videos_order(3))
%         disp(i);
%         disp('Att Chk 3 OK');
    else
        disp(i);
        disp('Att Chk 3 NOK');
    end
    
    for j = 3:5
        if ~strcmp(participant_trust_3{1, j}{1}, '')
            trust_prediction_3 = str2num(participant_trust_3{1, j}{1});
        end
    end
    
                                     
    participant_trust_predictions = [trust_prediction_1;
                                     trust_prediction_2;
                                     trust_prediction_3];
                                 
%% data concatenation...


obs_task_seq = [obs_task_seq; participant_observed_task];
pred_task = [pred_task; participant_prediction_task];
obs_task_sens_cap_seq = [obs_task_sens_cap_seq; participant_sensing_capabilities];
obs_task_proc_cap_seq = [obs_task_proc_cap_seq; participant_processing_capabilities];
pred_task_sens_cap = [pred_task_sens_cap; participant_sens_cap_pred_task];
pred_task_proc_cap = [pred_task_proc_cap; participant_proc_cap_pred_task];
trust_pred = [trust_pred; participant_trust_predictions];

    


    
                               
end