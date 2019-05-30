function [file_count, age] = input_count(age)
%% Input txt file number count

data_list = dir(sprintf('*Hand_IMU_%d*.txt', age));
file_count = numel(data_list);

end

