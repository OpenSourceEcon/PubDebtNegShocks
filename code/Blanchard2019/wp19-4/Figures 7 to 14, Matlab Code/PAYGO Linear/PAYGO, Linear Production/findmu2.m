function y=findmu2(miu)

global par kss calibration column index

%--Old CES version
%y = ( par.b .* kss(index).^(par.epsilon-1) .* (par.b .* kss(index).^par.epsilon + (1-par.b) ).^((1-par.epsilon)/par.epsilon) .*exp(miu+(par.sigma^2)/2) ) - calibration(1,index);

%--Cobb Douglas version
y = ( par.b .* kss(index).^(par.b-1) .*exp(miu+(par.sigma^2)/2) ) - calibration(1,index);

end
