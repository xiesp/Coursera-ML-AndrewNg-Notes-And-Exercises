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
    % 类似的,想要得到更新后的updated theta1,必须让 (X*theta - y) X的第二列
    % 所以很明显,我们转置X,X' 是1个2-m矩阵,和(X*theta - y)向后才能以后,
    % 是1个2-1的矩阵

    theta = theta - (alpha / m) * ( X' * (X*theta - y));
    
    % 注意,下面的是不对的,不需要求和函数了,矩阵乘法已经求出和了,不要理解错了
    % 和computeCost.m的比较一下就明白了
    %theta = theta - (alpha / m) * sum( X' * (X*theta - y));




    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
