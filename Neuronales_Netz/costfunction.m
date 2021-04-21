function [J grad] = costfunction(params, N, N_, num_labels, X, y, lambda)
%costfunction implements the neural network cost function for a two layer
% neural network which performs classification. In this case the mean
% square error function is used.
% It returns the cost function J and its gradient as a vector. params is a
% vector with all params of the neural network (weights). N and N_ are the 
% sizes of input and hidden layer. num_labels is the size of the output 
% layer. X are the input values and y are the desired output values.
% 

% Reshape params into matrices for forward propagation

omega1 = reshape(params(1:N_ * (N + 1)), N_, (N + 1));
omega2 = reshape(params((1 + (N_ * (N + 1))):end), num_labels, (N_ + 1));

% sets size of one sample
m = size(X, 1);

% sets memory for needed variables
J = 0;
omega1_grad = zeros(size(omega1));
omega2_grad = zeros(size(omega2));

% forward propagation
z2 = [ones(m, 1) X]*omega1';
a2 = activation_function(z2,1);
z3 = [ones(m, 1) a2] *omega2';
a3 = activation_function(z3,1);

y_mat = 1:num_labels == y;

% regularization summand
reg_j = lambda / (2*m) * (sum(omega1(:,2:end).^2,'all')+sum(omega2(:,2:end).^2,'all'));

% cost function 
J = (sum(-y_mat.*log(a3)-(1-y_mat).*log(1-a3),'all'))/m + reg_j;

% backpropagation
delta3 = a3 - y_mat;        % first set of deltas
% second set of deltas
delta2 = delta3 * omega2(:,2:end) .* grad_activation_function(z2,1);

% partial derivative of the cost function with respect to
D1 = delta2' * [ones(m,1) X];                           % w_ij^{1}
D2 = delta3' * [ones(m,1) a2];                          % w_ij^{2}

omega1_grad(:,2:end)=(D1(:,2:end) + lambda * omega1(:,2:end)) / m;
omega1_grad(:,1) = D1(:,1) / m;
omega2_grad(:,2:end)=(D2(:,2:end) + lambda * omega2(:,2:end)) / m;
omega2_grad(:,1) = D2(:,1) / m;

% convert gradient into one vector
grad = [omega1_grad(:) ; omega2_grad(:)];

end
