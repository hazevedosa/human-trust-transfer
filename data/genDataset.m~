clear all; close all;  clc;

raw_table = readtable('qualtricsRawData.xlsx');

num_responses = size(raw_table, 1);

for i = 1:num_responses
    participant_data = raw_table(i, :);
    participant_prediction_task = str2num(participant_data.randNumber{1});
    participant_videos_order_raw = participant_data.FL_15_DO{1};
    participant_videos_order = [str2num(participant_videos_order_raw(19)), ...
                                str2num(participant_videos_order_raw(39)), ...
                                str2num(participant_videos_order_raw(59)), ...
                                str2num(participant_videos_order_raw(79))];
                            
    participant_videos_order = setdiff(participant_videos_order, participant_prediction_task, 'stable');
    
    % sensing check
    
    
    % processing check
end