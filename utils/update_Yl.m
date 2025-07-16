function [Yl] = update_Yl(dia_M,M,X,P,G_l,Yu_sum,alpha,label_number,lambda1,num_cluster)
%UPDATE_YL �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
num_view = length(X);
for i=1:label_number
    b_i=zeros(1,num_cluster);
    final=zeros(1,num_cluster);
    for v=1:num_view
        b_i=b_i+alpha(v)^2*X{v}(:,i)'*M{v};
    end
    b_i=b_i+lambda1*G_l(:,i)'*P;
%     dia_M
%     b_i
%     ci
    final=dia_M-2*b_i;
    [min_value,min_ind]=min(final);
%     min_ind
    Yl(:,i)=zeros(num_cluster, 1);
    Yl(min_ind,i)=1;
end
% Yl
end

