function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    % 非向量化的方式
    % x=X(:,2);
    % theta0=theta(1);
    % theta1=theta(2);
    % theta0=theta0-alpha/m*sum(X*theta-y);
    % theta1=theta1-alpha/m*sum((X*theta-y).*x);
    % theta=[theta0;theta1];


    % 我们需要实现的是向量化的代码
    % 注意 X 的维度是m-2,theta维度是2-1
    % 那么(X * theta - y ) 维度是 m-1

    % 所以我们不能直接 (X*theta - y) * X
    % 如果我们想要得到的是更新后的theta0,我们必须让 (X*theta - y) 乘以X的第一列
    % 类似的,想要得到更新后的 theta1,必须让 (X*theta - y) X的第二列
    % 所以很明显,我们转置X,X'.shape =[2 m],X'的第一行是x0,第二行是x1...
    % 和(X*theta - y).shape = [m,1]相乘,

    theta = theta - (alpha / m) * ( X' * (X*theta - y));
    
  
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
