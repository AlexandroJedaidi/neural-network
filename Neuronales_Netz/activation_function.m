function f = activation_function(z,enum)
% computes activation function
% enum==1: SIGMOID function F(z)=1/(1+exp(-z))
if(enum==1)
    f = 1.0 ./ (1.0 + exp(-z));
end

end

