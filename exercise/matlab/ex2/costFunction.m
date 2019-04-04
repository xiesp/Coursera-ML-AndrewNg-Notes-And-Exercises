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


%  先计算出sigmoid函数的参数, theta.shape = [n+1 1],x.shape[m n+1]
%  z.shape = [m 1]
z = X * theta; 
g = sigmoid(z);

% 计算代价函数,注意y'.shaspe=[1 m],相当于求和
J = -1.0 / m * (( y' * log(g))  + (1- y)' * log(1 - g) ) 

% 需要求和公式的如下
% J= -1 * sum( y .* log( g ) + (1 - y ) .* log( (1 - g) ) ) / m ;

% 计算梯度,每一个theta-j的梯度
% (g-y).shape=[m 1],X'.shape=[n+1 m]
% so grad.shape=[n+1 1]
 grad = 1/m * ( X' * ( g - y ) );


% =============================================================

end
