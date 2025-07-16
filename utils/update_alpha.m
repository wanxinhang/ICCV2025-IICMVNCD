function [alpha] = update_alpha(cnt_1)
%UPDATE_ALPHA 此处显示有关此函数的摘要
%   此处显示详细说明
inv_rv2 = 1 ./ cnt_1; 

% 归一化计算 alpha
alpha = inv_rv2 / sum(inv_rv2);
end

