clear all;
clc;
warning off;
addpath(genpath('ClusteringMeasure'));
addpath(genpath('utils'));
addpath(genpath('measure'));
addpath(genpath('eval'));

% 文件保存路径
eachResSavePath = 'each_Res_select_version/';
eachACCSavePath = 'each_ACC_select_version/';
globalACCSavePath = 'global_select_version/';
globalACCSavePath_doc = 'global_select_doc/';

% 创建保存路径
if (~exist(eachResSavePath, 'file'))
    mkdir(eachResSavePath);
    addpath(genpath(eachResSavePath));
end
if (~exist(eachACCSavePath, 'file'))
    mkdir(eachACCSavePath);
    addpath(genpath(eachACCSavePath));
end

if (~exist(globalACCSavePath, 'file'))
    mkdir(globalACCSavePath);
    addpath(genpath(globalACCSavePath));
end

if (~exist(globalACCSavePath_doc, 'file'))
    mkdir(globalACCSavePath_doc);
    addpath(genpath(globalACCSavePath_doc));
end

% 数据集名称
datasetName = {'UCI-DIGIT'};
% 遍历数据集

for dataIndex = 1:length(datasetName)
    % 加载数据集
    % Load dataset
    load(['F:\wxh_work\datasets\MultiView_Dataset/',datasetName{dataIndex},'.mat' ]);
    datasetName{dataIndex}
    num_cluster = length(unique(Y));
    num_view = length(X);
    num_sample = size(X{1}, 1);
    
    % Normalize features
    for v = 1:num_view
        X{v} = zscore(X{v})';
    end
    
    % Adjust labels if necessary
    if length(find(Y == 0)) > 0
        for i = 1:size(X{1}, 2)
            Y(i) = Y(i) + 1;
        end
    end
    
    % Reorganize data by target classes
    target_classes = [1:floor(num_cluster / 2)];
    target_indices = ismember(Y, target_classes);
    non_target_indices = ~target_indices;

    Y_new = [Y(target_indices); Y(non_target_indices)];
    Gl = Y(target_indices);
    label_number = length(Gl);
    G_l = full(sparse(1:label_number, Gl, 1, label_number, num_cluster));
    clear Gl

    X_new = cell(size(X));
    for v = 1:length(X)
        X_v = X{v};
        X_new{v} = [X_v(:, target_indices), X_v(:, non_target_indices)];
    end
    clear X Y
    Y = Y_new;
    X = X_new;
    clear X_new Y_new

    ResBest = zeros(1, 8);
    ResStd = zeros(1, 8);

    % Parameter settings
    r1 = 10.^(0:1:5);
    r2 = 10.^(-1:1:5);

    % 初始化全局最佳结果追踪器
    global_max_ACC = -Inf;  % 全局最大 val_unlabeled(1)
    global_best_res = [];            % 对应的最佳结构体
    global_best_hyperparams = [];    % 对应的超参数
    global_max_val_unlabeled = -Inf * ones(1, 10); % 假设 val_unlabeled 有 3 项 (ACC, NMI, Purity)
    for lam1_iter = 1:length(r1)
        for lam2_iter = 1:length(r2)
            % 针对当前超参数的最小 obj 追踪器
            local_min_obj = Inf;     % 当前超参数组合下 obj 的最小值
            local_best_res = [];    % 对应的最佳结构体

            % 将 repeat 循环拆分为 4 次，每次 5 个 parfor
            for outer_iter = 1:4
                % 用于保存每次 parfor 的结果
                temp_results = cell(5, 1);
                parfor inner_iter = 1:5
                    % 初始化随机标签
                    YY=zeros(num_sample,1);
                    YY(1:label_number)=Y(1:label_number);
                    YY(label_number+1:end) = randi([floor(num_cluster / 2) + 1, num_cluster], num_sample-label_number, 1);

                    YY_ini1 = full(sparse(1:num_sample, YY, 1, num_sample, num_cluster))';

                    lambda1 = r1(lam1_iter)
                    lambda2 = r2(lam2_iter)

                    % 运行算法
                    tic;
                    [obj, alpha, pre_Y1] = my_alg(X, num_cluster, lambda1, lambda2, YY_ini1, G_l', label_number);
                    time = toc;

                    % 提取有效 obj 的最后一个值
                    last_nonzero_idx = find(obj ~= 0, 1, 'last');
                    effective_obj_value = obj(last_nonzero_idx);

                    pre_Y = zeros(size(pre_Y1, 2), 1);
                    for j = 1:size(pre_Y1, 2)
                        pre_Y(j) = find(pre_Y1(:, j), 1);
                    end

                    % Evaluate total performance
                    val_total = my_eval_y(pre_Y', Y);

                    % Evaluate unlabeled samples
                    Y_l = Y(1:label_number);
                    Y_u = Y(label_number + 1:end);
                    pre_Y_u = pre_Y(label_number + 1:end);

                    % Check if predictions fall into labeled space
                    for j = 1:length(pre_Y_u)
                        if ismember(pre_Y_u(j), unique(Y_l))
                            pre_Y_u(j) = randi([floor(num_cluster / 2) + 1, num_cluster]);
                        end
                    end

                    val_unlabeled = my_eval_y(pre_Y_u', Y_u);

                    % 保存当前结果
                    temp_results{inner_iter} = struct('time', time, 'obj', obj, 'alpha', alpha, ...
                        'pre_Y', pre_Y, 'target_classes', [], ...
                        'val_total', val_total, 'val_unlabeled', val_unlabeled, ...
                        'effective_obj_value', effective_obj_value);
                end

                % 合并 parfor 结果，更新当前超参数对应的最小 obj
                for inner_iter = 1:5
                    if temp_results{inner_iter}.effective_obj_value < local_min_obj
                        local_min_obj = temp_results{inner_iter}.effective_obj_value;
                        local_best_res = temp_results{inner_iter};
                    end
                end
            end

            % 保存每个超参数组合的最佳结果
            resFile = [eachResSavePath, datasetName{dataIndex}, '-lam1=', num2str(lam1_iter), ...
                '-lam2=', num2str(lam2_iter), '.mat'];
            save(resFile, 'local_best_res');
            resFile1 = [eachACCSavePath, datasetName{dataIndex}, '-lam1=', num2str(lam1_iter), ...
                '-lam2=', num2str(lam2_iter), '-ACC=', num2str(local_best_res.val_unlabeled(1)),'.mat'];
            save(resFile1, 'local_best_res');
            % 更新全局最佳 val_unlabeled(1)
            if local_best_res.val_unlabeled(1) > global_max_ACC
                global_max_ACC = local_best_res.val_unlabeled(1);
                global_best_res = local_best_res;
                global_best_hyperparams = [r1(lam1_iter), r2(lam2_iter)];
            end
            for k = 1:length(global_max_val_unlabeled)
                if local_best_res.val_unlabeled(k) > global_max_val_unlabeled(k)
                    global_max_val_unlabeled(k) = local_best_res.val_unlabeled(k);
                end
            end

        end
    end

    % 保存全局最佳结果
    globalMaxFile1 = [globalACCSavePath, datasetName{dataIndex}, '_GlobalMax.mat'];
    globalMaxFile2 = [globalACCSavePath_doc, datasetName{dataIndex}, '-ACC=', num2str(global_max_ACC), '.mat'];
    save(globalMaxFile1, 'global_best_res', 'global_best_hyperparams', 'global_max_val_unlabeled', 'global_max_ACC');
    save(globalMaxFile2, 'global_best_res', 'global_best_hyperparams', 'global_max_val_unlabeled', 'global_max_ACC');
end
