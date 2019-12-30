% function output=c2tD(epsi);
% Compute the expected consumption of old generation
function xxx=c2debtBARE(epsi)

global par index S transfer; 

%M is as defined in Equation(20) of the notes
M(1,:,index) = par.b .* (S(2,:,index).^par.epsilon) + (1-par.b);

%Rtplus1 is just R(t+1)
Rtplus1 = (par.b.*exp(epsi+par.mu(index))).*(S(2,:,index).^(par.epsilon-1)) .* M(1,:,index).^((1-par.epsilon)/par.epsilon);
C2 = Rtplus1.*S(2,:,index) + transfer(1,:,index);

xxx(:,1) = (C2);