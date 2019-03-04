function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%


%% for 循环版本  %%%
% num_points = zeros(1,K);
% % 循环每个数据
% for i = 1:m
% 	%找出第i个数据的中心点,累加第i个点的数据
% 	centroids(idx(i),:) = centroids(idx(i),:) + X(i,:);
% 	num_points(idx(i)) = num_points(idx(i)) + 1;
% end
% centroids
% % 循环求平均数
% for j = 1:K
% 	centroids(idx(j),:) = centroids(idx(j),:) ./ num_points(idx(j));
% end


% 快速版本
for j = 1:K
	% 1表示每一列的平均值
    centroids(j, :) = mean(X(idx==j, :), 1);
end


% =============================================================


end

