function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % Number of training examples

% You need to return the following variables correctly
p = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters. 
%               You should set p to a vector of 0's and 1's
%



% prob = sigmoid(X * theta);
% p( prob>=0.5 ) = 1;
% p( prob<0.5 ) = 0;

% 上面代码也是可以的

%其实最终就是判断有多少大于0.5,大于0.5的就看做是1,其他是0
p(sigmoid( X * theta) >= 0.5) = 1;


% =========================================================================


end
