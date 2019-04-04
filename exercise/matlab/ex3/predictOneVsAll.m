function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
% [10 1]
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
%[5000 1]
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       
% X是5000-401,y是401-10,X*all_theta'是一个5000-10矩阵,all_theta.shape=[10 401]
% 5000行当中,每一行都是对这一行预测的10个可能结果,也就是说,每一个数字都预测了10个结果
y_pred = sigmoid(X * all_theta');

% 寻找每一行的最大值和下标,因为最大值就表示预测的可能值是最大的
% max_num就是最大值,p是对应的下标
% 注意max最后一个参数2表示以行为单位运行max函数
% max_num,p都是 5000 -1的数组
[max_num, p] = max(y_pred, [], 2);






% =========================================================================


end
