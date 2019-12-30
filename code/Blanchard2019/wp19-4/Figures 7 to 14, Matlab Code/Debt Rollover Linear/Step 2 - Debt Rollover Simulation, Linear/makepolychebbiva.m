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


po_k    = ord(1);
po_a    = ord(2);

xk      = chebpol(po_k,x(:,1));
xa      = chebpol(po_a,x(:,2));
col = factorial(po_k+2)/(factorial(po_k)*factorial(2));
xmat = zeros(size(xk,1),col);
index = 1;
for kj  = 0:po_k
	for aj  = 0:po_a
             if kj + aj <= po_k
                 xmat(:,index) = xk(:,kj+1).*xa(:,aj+1);
                 index = index +1;
             end
         end
    end
end

% **********************************************************************

% **********************************************************************
