function p = predict(omega1, omega2, X)
%PREDICT is a function that looks at the output of the network and predicts
% the label of the output. predict(omega1, omega2, X) returns a vector
% where each value is the maximum of the output of each sample.

m = size(X,1);

% last forward pass to get Y
phi_1 = activation_function([ones(m, 1) X] * omega1',1);
phi_2 = activation_function([ones(m, 1) phi_1] * omega2',1);

[~, p] = max(phi_2, [], 2);
end

