function [alpha] = update_alpha(cnt_1)
%UPDATE_ALPHA �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
inv_rv2 = 1 ./ cnt_1; 

% ��һ������ alpha
alpha = inv_rv2 / sum(inv_rv2);
end

