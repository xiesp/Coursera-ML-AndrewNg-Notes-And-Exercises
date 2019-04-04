function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% 最终返回的idx里面,是每一个样本点距离最近中心店的下标
% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%


% 全部的样本数目
m = size(X,1);
% 对样本数目循环
for i = 1:m
	% 设置一个最小的距离
	min_dis = 1e9;
	% 循环当前的中心点
	for j = 1:K
		% 计算当期那样本点和每一个中心点的距离
		dis = norm(X(i,:) - centroids(j,:))^2;
		if dis < min_dis
			min_dis = dis;
			idx(i) = j;
		end
	end
end





% =============================================================

end

