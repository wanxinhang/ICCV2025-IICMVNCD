function [obj] = get_obj(alpha,cnt_1,P,Y_l,G_l,Y_u,num_smaple,label_number,num_view,lambda1,lambda2)
%GET_OBJ 此处显示有关此函数的摘要
%   此处显示详细说明
obj=0;
for v=1:num_view
    obj=obj+alpha(v)^2*cnt_1(v);
end
% obj
obj=obj+lambda1*sum(sum((P*Y_l - G_l).^2));
% lambda1*sum(sum((P*Y_l - G_l).^2))
obj2=0;
for i=1:label_number
    for j=1:num_smaple-label_number
        obj=obj-lambda2*sum((G_l(:,i)-P*Y_u(:,j)).^2);
        obj2=obj2-lambda2*sum((G_l(:,i)-P*Y_u(:,j)).^2);
    end
end
% obj2
end

