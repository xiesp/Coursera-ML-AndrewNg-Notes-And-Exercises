function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
% 5000
m = size(X, 1);
% 400
n = size(X, 2);

% You need to return the following variables correctly 
% 10 - 401的参数矩阵,每一行都对应一个逻辑回归
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
% 在X前面添加一列1
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda. 
%
% Hint: theta(:) will return a column vector.
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell you
%       whether the ground truth is true/false for this class.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%     
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
%


% options = optimset('GradObj', 'on', 'MaxIter', 50);
% lambda = 1.0;

% % for循环版本的实现
% for c = 1:num_labels
% 	% 当前的y标签
% 	cur_y = (y == c);
% 	cur_theta = zeros(n+1,1);
% 	[cur_theta, cur_cost]  = fminunc(@(t)(lrCostFunction(t, X, cur_y,lambda)), cur_theta, options);
% 	all_theta(c,:) = cur_theta;
% end


for c = 1:num_labels
	% [401 1]
    initial_theta = zeros(n + 1, 1);
    options = optimset('GradObj', 'on', 'MaxIter', 50);
    % 注意,y实际上还是0/1,因为y==c得到就是y = 0或者1
    [theta] = ...
         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
                 initial_theta, options);
    % 每一行作为一个预测的参数
    all_theta(c, :) = theta(:)';
end




% =========================================================================


end
