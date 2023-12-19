import numpy as np

def is_valid_input(c, A_lt, b_lt, A_eq, b_eq, A_gt, b_gt):
    '''
    Checks inputs are valid
    '''
    if type(c) != np.ndarray: # Types are right
        raise TypeError("c should be a numpy array")
    if A_lt is None and A_eq is None and A_gt is None:
        raise ValueError("At least one of A_lt, A_eq, and A_gt should be defined")
    if A_lt is not None or b_lt is not None: # If we have A_lt or b_lt
        if type(A_lt) != np.ndarray or type(b_lt) != np.ndarray: # Types are right
            raise TypeError("A_lt and b_lt should both be None or numpy arrays")
        if len(b_lt.shape) != 1 or len(A_lt.shape) > 2:
            raise ValueError("Wrong number of dimensions for A_lt and b_lt")
        if c.shape[0] != A_lt.shape[1]: # Same number of decision variables in A and c
            raise ValueError("Number of decision variables doesn't match A_eq")
        if b_lt.shape[0] != A_lt.shape[0]: # Same number of constraints in A and b
            raise ValueError("Number of constraints doesn't match in A_lt and b_lt")
    if A_eq is not None or b_eq is not None: # If we have A_eq or b_eq
        if type(A_eq) != np.ndarray or type(b_eq) != np.ndarray: # Types are right
            raise TypeError("A_eq and b_eq should both be None or numpy arrays")
        if len(b_eq.shape) != 1 or len(A_eq.shape) > 2:
            raise ValueError("Wrong number of dimensions for A_eq and b_eq")
        if c.shape[0] != A_eq.shape[1]: # Same number of decision variables in A and c
            raise ValueError("Number of decision variables doesn't match A_eq")
        if b_eq.shape[0] != A_eq.shape[0]: # Same number of constraints in A and b
            raise ValueError("Number of constraints doesn't match in A_eq and b_eq")
    if A_gt is not None or b_gt is not None:
        if type(A_gt) != np.ndarray or type(b_gt) != np.ndarray: # Types are right
            raise TypeError("A_gt and b_gt should both be None or numpy arrays")
        if len(b_gt.shape) != 1 or len(A_gt.shape) > 2:
            raise ValueError("Wrong number of dimensions for A_gt and b_gt")
        if c.shape[0] != A_gt.shape[1]: # Same number of decision variables in A and c
            raise ValueError("Number of decision variables doesn't match A_gt")
        if b_gt.shape[0] != A_gt.shape[0]: # Same number of constraints in A and b
            raise ValueError("Number of constraints doesn't match in A_gt and b_gt")


def shift_negatives(A_1, b_1, A_2, b_2):
    '''
    Shift any negative b_2 to positive b_1
    '''
    if A_2 is not None and np.any(b_2 < 0): # Move any negative values to gt
        i = np.where(b_2 < 0)[0]
        if A_1 is None: # Make A_1 and b_1
            A_1, b_1 = -A_2[i], -b_2[i]
        else:
            A_1 = np.row_stack((A_1, -A_2[i]))
            b_1 = np.concatenate((b_1, -b_2[i]))
        A_2, b_2 = np.delete(A_2, i, axis=0), np.delete(b_2, i)
    
    if A_1 is not None: # Remove duplicated rows
        Ab_1 = np.column_stack((A_1, b_1))
        A_1 = np.unique(Ab_1, axis=0)[:,:-1]
        b_1 = np.unique(Ab_1, axis=0)[:,-1]

    return A_1, b_1, A_2, b_2


def shift_equality(A_eq, b_eq, A_lt, b_lt, A_gt, b_gt):
    '''
    Shift any constraints that are in both in the less than and greater
    than constriant into the equality constraint, makes the equality
    constraint all positive, and removes identical constraints
    '''
    if A_lt is not None and A_gt is not None: # Combine identical lt and gt constraints to eq
        Ab_lt = np.column_stack((A_lt, b_lt))
        Ab_gt = np.column_stack((A_gt, b_gt))
        
        i_lt = (Ab_lt[:, None] == Ab_gt).all(-1).any(-1) # Indexes of repeates (in lt)
        i_gt = (Ab_gt[:, None] == Ab_lt).all(-1).any(-1)

        if i_lt.any():
            # Move to equals matricies
            if A_eq is None:
                A_eq, b_eq = A_lt[i_lt,:], b_lt[i_lt]
            else:
                A_eq = np.row_stack(A_eq, A_lt[i_lt,:])
                b_eq = np.concatenate(b_eq, b_lt[i_lt])

            # Remove from intital matricies
            if b_lt.shape[0] == sum(i_lt):
                A_lt, b_lt = None, None
            else:
                A_lt, b_lt = A_lt[np.invert(i_lt),:], b_lt[np.invert(i_lt)]
            if b_gt.shape[0] == sum(i_gt):
                A_gt, b_gt = None, None
            else:
                A_gt, b_gt = A_gt[np.invert(i_gt),:], b_gt[np.invert(i_gt)]
    
    if A_eq is not None: 
        # Make equalities positive
        i_negb = np.where(b_eq < 0)[0]
        b_eq[i_negb] = -b_eq[i_negb]
        A_eq[i_negb,:] = -A_eq[i_negb,:]

        # Remove identical rows
        Ab_eq = np.column_stack((A_eq, b_eq))
        A_eq = np.unique(Ab_eq, axis=0)[:,:-1]
        b_eq = np.unique(Ab_eq, axis=0)[:,-1]
    
    return A_eq, b_eq, A_lt, b_lt, A_gt, b_gt


def l(b):
    '''
    Length of the array. 0 if None
    '''
    return 0 if b is None else b.shape[0]


def pick_basic(tab, basics):
    '''
    Picks the currenly non-basic variable to switch to a basic one.

    Chooses the first negative value in the maximization row in the
    table (Bland's Rule).
    '''
    w = tab.shape[1]-1 # width of the array
    options = np.setdiff1d(np.arange(w), basics) # non-basic variables
    return options[np.where(tab[0, options] < 0)[0][0]] # pick the first option less than 0



def ratio_test(tab, i):
    '''
    Performs a ratio test to find the limiting constraint and pick
    the basic variable to switch to a non-basic one.

    If two variables have the same ratio, it picks the first one
    (Bland's Rule).
    '''
    ratios = tab[1:, -1]/tab[1:, i] # Get ratios
    valid_i = np.where(tab[1:, i]>0)[0] # Indexes with positive values (Check in ratio test)
    j = valid_i[np.where(ratios[valid_i] == min(ratios[valid_i]))[0][0]] + 1 # Pick the valid index with the smallest ratio
    return j


def row_opperations(tab, i, j):
    '''
    Performs row operations to remove the new basic variable from all
    constraints and the objective except the one at row j.
    '''
    tab[j, :] = tab[j, :] / tab[j, i] # Make spot j, i 1
    for k in range(tab.shape[0]):
        if k == j:
            continue
        tab[k, :] = tab[k, :] - tab[k, i]*tab[j, :]


def get_x(tab, basics, d):
    '''
    Finds values in the x vector.
    
    Equal to the RHS Column value if it's a basic variable column, and 0
    otherwise
    '''
    x = np.zeros(tab.shape[1]-1) 
    x[basics] = tab[:, -1]
    return x[1:1+d]


def simplex(c: np.ndarray, # Maximization function
            A_lt: np.ndarray = None, b_lt: np.ndarray = None, # Ax<=b
            A_eq: np.ndarray = None, b_eq: np.ndarray = None, # Ax=b
            A_gt: np.ndarray = None, b_gt: np.ndarray = None, # Ax>=b
            min: bool = False) -> np.ndarray:
    '''
    Implementation of the Simplex Method to find a vector x that
    maximizes t(c)x given
        A_lt x <= b_lt
        A_eq x = b_eq
        A_gt x >= b_gt

    Uses a tableu implementation of the Simplex Method with the
    Two-Phase Method used to deal with degenerate initializations
    and Bland's Rule to stop cycling.

    :param c: Coefficients along the maximimzation function
    :param A_lt: Coefficents in the LHS of a less than or equal to 
    inequality.
    :param b_lt: RHS of the a less than or equal to problem.
    :param A_eq: Coefficents in the LHS of an equality
    :param b_eq: RHS of an equality
    :param A_gt: Coefficents in the LHS of a greater than or equal to 
    inequality.
    :param b_gt: RHS of the a greater than or equal to problem.
    :param min: 
    :return: The vector x of the optimized solution. If the solution
    is unbounded one of the values will be 'inf' and if no feasable
    region exists, the vector will have NaN as all its values.
    '''
    is_valid_input(c, A_lt, b_lt, A_eq, b_eq, A_gt, b_gt)

    if min:
        c = -c

    d = c.shape[0] # number of decision variables
    
    ## Setup
    if (A_lt is not None and np.any(b_lt < 0)) or A_eq is not None or (A_gt is not None and np.any(b_gt >= 0)): # Do Phase 1
        # Shift Around Arrays
        A_lt, b_lt, A_gt, b_gt = shift_negatives(A_lt, b_lt, A_gt, b_gt)
        A_gt, b_gt, A_lt, b_lt = shift_negatives(A_gt, b_gt, A_lt, b_lt)
        A_eq, b_eq, A_lt, b_lt, A_gt, b_gt = shift_equality(A_eq, b_eq, A_lt, b_lt, A_gt, b_gt)
        n_lt, n_eq, n_gt = l(b_lt), l(b_eq), l(b_gt)
        n = n_lt + n_gt # number of slack variables
        a = n_eq + n_gt # number of additional variables

        # Setup phase 1 problem
        tab = np.concatenate((np.ones(1), np.zeros(d+n), np.ones(a), np.zeros(1)))
        if A_lt is not None:
            cons_lt = np.column_stack((np.zeros(n_lt), A_lt, np.identity(n_lt), np.zeros((n_lt, n_gt+a)), b_lt))
            tab = np.row_stack((tab, cons_lt))
        if A_eq is not None:
            cons_eq = np.column_stack((np.zeros(n_eq), A_eq, np.zeros((n_eq, n)), np.identity(n_eq), np.zeros((n_eq, n_gt)), b_eq))
            tab = np.row_stack((tab, cons_eq))
        if A_gt is not None:
            cons_lt = np.column_stack((np.zeros(n_gt), A_gt, np.zeros((n_gt, n_lt)), -np.identity(n_gt), np.zeros((n_gt, n_eq)), np.identity(n_gt), b_gt))
            tab = np.row_stack((tab, cons_lt))
        basics = np.concatenate((np.zeros(1, dtype=np.int64), np.arange(d+1, d+n_lt+1), np.arange(d+n+1, d+n+a+1)))

        # Simplify additional variables
        for i in range(n_lt+1, n+n_eq+1):
            tab[0,:] -= tab[i,:]
        
        ## Simplex method
        while np.any(tab[0,1:-1] < 0): # Termination criteria
            i = pick_basic(tab, basics)

            # Make sure theres a bound
            if np.all(tab[1:, i] <= 0): # Unbounded solution
                x = get_x(tab, basics, d)
                for k in range(1,len(x)):
                    if k == i:
                        x[k-1] = np.inf
                    if sum(tab[:, k] == 0) == n-1 and sum(tab[:, k] == 1) == 1: # Variable is basic
                        if tab[k-1, i] != 0: 
                            x[k-1] = np.inf
                return x

            j = ratio_test(tab, i)

            row_opperations(tab, i, j)
            basics[j] = i # Put i in the basics matrix

        # Check feasible region
        if tab[0, -1] != 0: # Check optimal value
            return np.full(d, np.NaN)
        
        # Setup phase 2 problem
        tab = np.delete(tab, np.s_[d+n+1:n+d+a+1], 1)
        tab[0, 1:d+1] = -c
        for i in range(1,d+1):
            if i in basics: # Variable is basic
                j = np.where(tab[1:, i] == 1)[0][0] + 1
                tab[0, :] -= tab[0, i]*tab[j, :]
        tab = tab[np.where(basics < d+n+1)[0],:]
        basics = basics[np.where(basics < d+n+1)[0]]
        
    else: # Setup phase 2 problem
        A_lt, b_lt, A_gt, b_gt = shift_negatives(A_lt, b_lt, A_gt, b_gt)
        n = b_lt.shape[0]

        # Build tab
        obj = np.concatenate((np.ones(1), -c, np.zeros(n+1)))
        cons = np.column_stack((np.zeros(n), A_lt, np.identity(n), b_lt))
        tab = np.row_stack((obj, cons))
        basics = np.arange(d, tab.shape[1]-1)
        basics[0] = 0
    
    ## Simplex method
    while np.any(tab[0,1:-1] < 0): # Termination criteria
        i = pick_basic(tab, basics)

        # Make sure theres a bound
        if np.all(tab[1:, i] <= 0): # Unbounded solution
            x = get_x(tab, basics, d)
            for k in range(1, len(x)+1):
                if k == i:
                    x[k-1] = np.inf
                if k in basics: # Variable is basic
                    if tab[np.where(basics==k)[0]-1, i] != 0: 
                        x[k-1] = np.inf
            return x

        j = ratio_test(tab, i)

        row_opperations(tab, i, j)
        basics[j] = i # Put i in the basics matrix

    return get_x(tab, basics, d)