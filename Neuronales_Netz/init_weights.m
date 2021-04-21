function W = init_weights(N, N_)
%RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with size
% N and its neighbour layer with size N_.

% set range for weights
eps = sqrt(6)/(sqrt(N+N_));

% set weights uniformly in range [-eps,eps]
W = rand(N_,1+N)*2*eps-eps;

end
