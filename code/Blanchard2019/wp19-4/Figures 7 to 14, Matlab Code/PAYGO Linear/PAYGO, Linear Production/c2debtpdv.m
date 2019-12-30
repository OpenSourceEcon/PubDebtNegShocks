% function output=c2tD(epsi);
% Compute the expected consumption of old generation
function xxx=c2debtpdv(epsi)

global par index S transfer c_d kt_debt z; 

%M is as defined in Equation(20) of the notes
M = par.b .* (kt_debt(z,c_d,index).^par.epsilon) + (1-par.b);

%Rtplus1 is just R(t+1)
Rtplus1 = (par.b.*exp(epsi+par.mu(index))).*(kt_debt(z,c_d,index).^(par.epsilon-1)) .* M.^((1-par.epsilon)/par.epsilon);
C2 = Rtplus1.*kt_debt(z,c_d,index) + transfer(1,1,index);

xxx(:,1) = (C2).^(1-par.gamma(index));
