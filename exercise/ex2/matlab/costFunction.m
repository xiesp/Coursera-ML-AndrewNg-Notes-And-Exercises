function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%


%  先计算出sigmoid函数的参数, theta 是(n+1)-1维,x是 m - (n+1)维
z = X * theta; 
g = sigmoid(z);

% 计算代价函数,注意y'已经是1 - m维度,相当于求和了
J = 1 / m * ( - ( y' * log(g))   - (1- y)' * log(1 - g) ) 

% 需要求和公式的如下
% J= -1 * sum( y .* log( g ) + (1 - y ) .* log( (1 - g) ) ) / m ;

% 计算梯度,注意也不用sum,矩阵乘法已经求和了
 grad = 1/m * ( X' * ( g - y ) );




% =============================================================

end
