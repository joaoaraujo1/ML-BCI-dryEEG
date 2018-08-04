function [ PDMatrix ] = P( Matrix )
%P Operator to make symmetric matrices be positive definite
% Joao Araujo 2017

[U,S,V] = svd(Matrix);
PDMatrix = U * abs(S) * V';


end

