function [one_hot] = inintialize_Y(X,k)
%ININTIALIZE_Y 此处显示有关此函数的摘要
%   此处显示详细说明
n=size(X{1},2);
num_view=size(X,1);
% XX=cell(num_view,1);
% U = cell(num_view,1);
% V = cell(num_view,1);
% for v=1:num_view
%     [U{v},~,V{v}] = svd(X{v}','econ');
%     XX{v} = U{v}*V{v}';
% end
M=X{1}';
for v=2:num_view
    %         X{v} = zscore(X{v})';
    M=[M,X{v}'];
end
[idx, ~] = kmeans(M, k);
% 创建一个 n x k 的 one-hot 编码矩阵
one_hot = zeros(n, k);
for i = 1:n
    one_hot(i, idx(i)) = 1; % 将聚类标签转换为 one-hot 编码
end
end

