% xmat=makepolycheb(ord,x);
%
%            For the (Tx2) matrix x and the (1x2) vector ord
%            this creates a [T x ord(1)*ord(2)] matrix where 
%            each row contains the ord(1)*ord(2) bivariate Chebyshev 
%            polynomial terms.
%
%		November 9 1998
%
% ------------------------------------------------------------------

function xmat=makepolychebbiva(ord,x)

po_a    = ord(1);
po_d    = ord(2);

xa      = chebpol(po_a,x(:,1)); %This returns the chebyshev-appropriate values of the various terms of "At", which acts like the "x" in a normal polynomial; i.e. if this were a function for a regular polynomial of degree 2, it would return 1, At, and At^2. 
xd      = chebpol(po_d,x(:,2)); %This returns the chebyshev-appropriate values of the various terms of "Dt", which acts like the "x" in a normal polynomial; i.e. if this were a function for a regular polynomial of degree 2, it would return 1, Dt, and Dt^2.

numvar=2;
col = factorial(po_a+numvar)/(factorial(po_a)*factorial(numvar));
xmat = zeros(size(xa,1),col);

index = 1;
for aj  = 0:po_a
	for dj  = 0:po_d
        if aj + dj <= po_a
                 xmat(:,index) = xa(:,aj+1).*xd(:,dj+1);
                 index = index +1;
        end 
    end
end

end

% **********************************************************************

% **********************************************************************
