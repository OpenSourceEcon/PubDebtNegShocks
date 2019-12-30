function y=findepsilon(miu)

global par kss R
y = ( par.b * kss^(par.epsilon-1) * (par.b * kss^par.epsilon + (1-par.b) )^((1-par.epsilon)/par.epsilon) *exp(miu+(par.sigma^2)/2) ) - R;



end
