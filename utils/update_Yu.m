function [Yu] = update_Yu(dia_M,M,X,Gl_sum,G_l,P,alpha,num_smaple,label_number,lambda2,num_cluster)
%UPDATE_YU 此处显示有关此函数的摘要
%   此处显示详细说明
num_view = length(X);
for i=1:num_smaple-label_number
    b_i=zeros(1,num_cluster);
    c_i=zeros(1,num_cluster);
    final=zeros(1,num_cluster);
    for v=1:num_view
        b_i=b_i+alpha(v)^2*X{v}(:,i+label_number)'*M{v};
    end
    ci=lambda2*Gl_sum'*P;
    final=dia_M-2*b_i+2*ci;
    [min_value,min_ind]=min(final);
    Yu(:,i)=zeros(num_cluster, 1);
    Yu(min_ind,i)=1;
end

end
