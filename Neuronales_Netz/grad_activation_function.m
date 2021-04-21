function f = grad_activation_function(z,enum)
% grad_activation_function returns the gradient of the actual activation
% function. 
% enum == 1: \nabla f(z)= sigmoid(z)*(1-sigmoid(z)).
f = zeros(size(z));

if(enum==1)
    f = activation_function(z,1) .* (1 - activation_function(z,1));
end

end

