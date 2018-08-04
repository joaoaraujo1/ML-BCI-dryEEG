function [model_lda] = ldaTrain(Data,lambda)
% LDATRAIN - Train an LDA model using least squares approach with a
% regularization parameter
%
%   João Araújo, 2018
%

X_up = Data.UP;
X_down = Data.DOWN;

model_lda.w = ((mean(X_up)-mean(X_down))/( ((1-lambda)*cov(X_down)+ lambda * eye(length(cov(X_down)))) ...
    + ( (1-lambda) * cov(X_up) + lambda * eye(length(cov(X_up))))))';

model_lda.b = (mean(X_down)+mean(X_up))*(-model_lda.w)/2;

end
