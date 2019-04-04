function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)



%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network

% 传入的input_layer_size,hidden_layer_size是没有加上bias term数目的,是400,25
% 但是nn_param里面的是包括了偏差单元的,所以注意下面公式计算的时候的+1

% 注意这里的计算公式,其实就是计算参数矩阵大小的公式
% 恢复参数矩阵

% Theta1.shpae = [25 401]
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

% Theta2.shpae = [10 26]
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m


% y是5000-1的向量,我们需要先形成5000-10的Y矩阵,作为神经网络需要对比的正确输出
% 先全部初始化为0
Y = zeros(m,num_labels);
% 循环5000次
for i=1:m
	%第i行的1-10向量的第y(i)个值变为1
	Y(i,y(i)) = 1;
end


% Theta1.shape = [25, 400+1], Theta2.shape = [10, 25+1]
% 为X增加1列1,X:5000-401
X = [ones(m, 1) X];

% 开始向前传播的计算
a1 = X;
% 注意需要转置,z2:5000-25
z2 = a1 * Theta1';
% sigmoid化z3
a2 = sigmoid(z2);
% 为a2增加一列1,现在a2是5000-26
a2 = [ones(m,1) a2];
% 计算z3
z3 = a2 * Theta2';
% 计算a3,a3:5000-10
a3 = sigmoid(z3);

% a3就是我们的预测的值
y_pred = a3;

%%%%% 计算J  %%%%%

% 注意Y是5000-10,y-pred也是5000-10,注意是点乘!!!!!!!!!!
% J现在还是5000-10
% 向量乘法一次性完成了所有的计算
% 注意,这里矩阵乘法不行!!!!必须是点乘,同下标元素之间的相乘!!!!好好理解一下!!!!
J= Y.*log(y_pred) + (1-Y) .* log(1-y_pred);
% 2次sum求出所有元素的和
J = -1/ m * sum(sum(J));



%%%%% 增加正则化部分 %%%%%

% 注意用排除掉第一列,也就是列下标为0的那一列
theta1Square = Theta1(:,2:end) .^ 2;
theta2Square = Theta2(:,2:end) .^ 2;
J = J + lambda / (2.0 * m)  * (  sum(sum(theta1Square))  + sum(sum(theta2Square)) );




%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

% [5000-10],注意这里是矩阵,笔记里面推导的delta应该是向量,那是因为
% 这里是5000个样本一起计算,所以产生5000个10个元素的行向量
% 注意好好理解一下!
delta3 = a3 - Y;


% 现在要从delta3推导得到delta2,Theta1.shape = [25, 400+1], Theta2.shape = [10, 25+1]
% 首先确定delta2的大小,因为m=5000,而这一层激活单元是25个,所以delta2应该是[5000 25],这表示
% 5000个样本里面,每一个样本,都输出了25个z2的值的预测值
% 注意笔记公式证明部分,delta2的5000行里面的每一行,其实对应的是笔记里面的一个delta2向量(笔记里面是列向量)
% 我们现在假设是第i行,这个向量是delta2(i).那么delta(i)其实就是delta3(i)向量,
% 右乘theta2矩阵转置
% 所以下面把delta3矩阵转置,变成[10 - 5000],这样每一列表示的是一个特定样本,对10个标签的误差
% 然后这个转置矩阵,右乘Theta2转置

% [25 10]  *  [10 - 5000] = [25 5000]
% 再次转置,变成[5000 25],以便和[5000 25]的2点乘
% [5000-25]的delta2矩阵
delta2 = (Theta2(:,2:end)' *  delta3')' .* sigmoidGradient(z2);

% 没有delta1

% 计算梯度矩阵

% [10 5000] * [5000 26] = [10- 26]
Delta2 = delta3' * a2; 

% [25 5000] * [5000 401] = [25 401]
Delta1 = delta2' * a1; 


Theta1_grad = Delta1 /m;
Theta2_grad = Delta2 /m;




% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% 不应该正则化第一列,也就是bias unit
Theta1(:,1) = 0;  
Theta2(:,1) = 0; 
Theta1_grad = Theta1_grad + lambda/m * Theta1;
Theta2_grad = Theta2_grad + lambda/m * Theta2;




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
