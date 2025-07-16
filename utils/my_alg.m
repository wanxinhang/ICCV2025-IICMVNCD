function [obj,alpha,Y] = my_alg(X,num_cluster,lambda1,lambda2,Y,G_l,label_number)
%MY_ALG 此处显示有关此函数的摘要
%   此处显示详细说明
maxIter=100;
num_view = length(X);
num_smaple=size(X{1},2);
obj=zeros(100,1);
%initialize A,Z,P of each view as 'identity matrix'
%conduct k-means of the first view to initialize Y
W=cell(num_view,1);
A=cell(num_view,1);
P=eye(num_cluster);;%num of samples in each cluster
M=cell(num_view,1);
alpha = ones(num_view,1)/num_view;
for v=1:num_view
    W{v}=zeros(size(X{v},1),num_cluster);
    A{v}=rand(num_cluster,num_cluster);
    for i=1:num_cluster
       W{v}(i,i)=1;
    end
end
%initialize Y
% end
flag = 1;
iter = 0;
while flag
    iter=iter+1
    
    
    W=update_W(X,Y,A);
    A=update_A(W,X,Y);
    Y_u = Y(:, label_number+1:end);
    sum_M=zeros(num_cluster,num_cluster);
    for v=1:num_view
        M{v}=W{v}*A{v};
        sum_M=sum_M+alpha(v)^2*M{v}'*M{v};
    end
    
    dia_M = diag(sum_M)';

    Yu_sum = sum(Y_u, 2);

    Y_l=update_Yl(dia_M,M,X,P,G_l,Yu_sum,alpha,label_number,lambda1,num_cluster);
    Gl_sum = sum(G_l, 2);
    Y_u=update_Yu(dia_M,M,X,Gl_sum,G_l,P,alpha,num_smaple,label_number,lambda2,num_cluster);
    Y=[Y_l,Y_u];
    cnt_1=zeros(num_view,1);
    for v=1:num_view
        cnt_1(v) = sum(sum((X{v} - W{v}*A{v}*Y).^2));
    end
    alpha=update_alpha(cnt_1);

    obj(iter)=get_obj(alpha,cnt_1,P,Y_l,G_l,Y_u,num_smaple,label_number,num_view,lambda1,lambda2);
    if (iter>2) && (abs((obj(iter)-obj(iter-1))/(obj(iter)))<1e-5 || iter>maxIter)
        flag =0;
    end
    
end
end

