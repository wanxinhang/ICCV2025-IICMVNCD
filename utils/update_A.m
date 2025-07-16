function [A] = update_A(W,X,Y)
%UPDATE_Z 此处显示有关此函数的摘要
%   此处显示详细说明
num_view=size(X,1);
A = cell(num_view,1);
YtY = Y * Y'; % 计算 Y * Y'

% 找到对角线元素为 0 的位置
diag_indices = 1:size(YtY, 1); % 对角线索引
diag_elements = diag(YtY); % 获取对角线元素
zero_diag_indices = (diag_elements == 0); % 找到对角线元素为 0 的位置

% 用一个小正数替换这些对角线元素
epsilon = 1e-5; % 预设小正数
YtY(diag_indices(zero_diag_indices), diag_indices(zero_diag_indices)) = epsilon;

for v=1:num_view
%     Y * Y'
    A{v} = (W{v}' * (X{v} * Y'))* inv(YtY);
end
end

