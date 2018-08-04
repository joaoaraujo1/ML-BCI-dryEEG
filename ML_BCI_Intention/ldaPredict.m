function [predictions] = ldaPredict(Data,model_lda)
% LDAPREDICT - Apply a previously trained LDA model to the data and output
% the prediction vector
%
%   João Araújo, 2018
%

X = [Data.UP;Data.DOWN];
y = [ones(size(Data.UP,1),1);-1 * ones(size(Data.DOWN,1),1)];

for p_idx = 1:size(X,1)
    predictions(p_idx) = sign(model_lda.w' * X(p_idx,:)'  + model_lda.b);
end

end
