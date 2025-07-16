function [W] = update_W(X,Y,A)
%UPDATE_Z 此处显示有关此函数的摘要
%   此处显示详细说明
num_view=size(X,1);
W = cell(num_view,1);
T = cell(num_view,1);
U = cell(num_view,1);
V = cell(num_view,1);
% Y+ 1e-1
for v=1:num_view
    T{v} = X{v}*(Y)'*A{v}';
%     X{v}*(Y)'*A{v}'
    [U{v},~,V{v}] = svd(T{v},'econ');
    W{v} = U{v}*V{v}';
end
end

